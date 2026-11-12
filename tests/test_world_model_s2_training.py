"""Check S2 current-turn masks, exposure accounting, and step validation timing."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from runtime.world_model import s2_training as training
from runtime.world_model import s2_loss_metrics as metrics
from runtime.world_model_training import consumed_training_index_plan

ROOT = Path(__file__).resolve().parents[1]


def exposure_args(count=252160):
    """Use the actual production schedule with synthetic unique row identities."""
    plan = consumed_training_index_plan(count, total_steps=1371, num_train_epochs=1,
        replica_count=184, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        seed=900913, preserve_order=False)
    return dict(run_id='run', method_id='method', run_config_sha256='a'*64,
        corpus_manifest_sha256='b'*64, train_sha256='c'*64,
        tokenizer_arm_manifest_sha256='d'*64, tokenizer_manifest_sha256='e'*64,
        record_ids=[f'row-{i}' for i in range(count)], fact_ids=[f'fact-{i}' for i in range(count)],
        question_families=['S2.1']*count, prompt_form_ids=['train']*count,
        source_stages=['S2']*201728 + ['S1']*37824 + ['S0']*12608,
        consumed_indices=plan['logical_indices'], distributed_padding_indices=plan['distributed_padding_indices'],
        seed=900913, total_steps=1371, num_train_epochs=1, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, data_parallel_size=184, preserve_order=False,
        padding_policy=plan['padding_policy'], exposure_scope='all_eligible_train_rows',
        dataset_id=training.S2_DATASET_ID, family_contract_routes_sha256='f'*64)


def test_full_exposure_declares_only_104_padding_repeats_and_all_unique_rows():
    args=exposure_args()
    receipt=training.build_training_exposure_manifest(**args,status='complete',completed_global_step=1371)
    assert receipt['logical_exposure']['record_occurrences']==252160
    assert receipt['logical_exposure']['unique_record_count']==252160
    assert receipt['logical_exposure']['all_eligible_train_rows_exposed'] is True
    assert receipt['physical_exposure']['record_occurrences']==252264
    assert receipt['physical_exposure']['distributed_padding']['record_occurrences']==104
    assert receipt['logical_exposure']['source_stage_counts']=={'S2':201728,'S1':37824,'S0':12608}
    assert receipt['schedule']['global_batch_size']==184


@pytest.mark.parametrize('mutation', ['incomplete','duplicate_identity','nontrain','unknownstage','badcompletion'])
def test_complete_exposure_rejects_false_completeness(mutation):
    args=exposure_args()
    if mutation=='incomplete': args['consumed_indices']=args['consumed_indices'][:-1]
    elif mutation=='duplicate_identity': args['record_ids'][0]=args['record_ids'][1]
    elif mutation=='nontrain': args['prompt_form_ids'][0]='test'
    elif mutation=='unknownstage': args['source_stages'][0]='S3'
    with pytest.raises(ValueError):
        training.build_training_exposure_manifest(**args,status='complete',
            completed_global_step=1370 if mutation=='badcompletion' else 1371)


def test_snapshot_exposure_is_a_real_prefix_not_full_training():
    args=exposure_args()
    args['consumed_indices']=args['consumed_indices'][:200*184]
    args['distributed_padding_indices']=[]
    receipt=training.build_training_exposure_manifest(**args,status='snapshot',completed_global_step=200)
    assert receipt['logical_exposure']['record_occurrences']==36800
    assert receipt['logical_exposure']['all_eligible_train_rows_exposed'] is False
    assert receipt['physical_exposure']['distributed_padding']['record_occurrences']==0


@pytest.mark.parametrize('metadata,expected', [
    ({'question_family':'S2.13','catalog_case_id':'S2.13.40'}, ('stage:S2','family:S2.13','source_trace:S2.13.40')),
    ({'source_stage':'S0','question_family':'human_symbol_to_ensembl','source_trace':'S0.1'},
     ('stage:S0','family:human_symbol_to_ensembl','source_trace:S0.1')),
    ({'source_stage':'S1','question_family':'layer_tag_parsing','source_trace':'S1.6.1'},
     ('stage:S1','family:layer_tag_parsing','source_trace:S1.6.1'))])
def test_loss_groups_separate_s2_from_original_replay(metadata,expected):
    assert tuple(metrics.LOSS_GROUP_NAMES[i] for i in metrics.loss_group_ids(metadata))==expected


def test_unknown_case_loss_group_is_rejected():
    with pytest.raises(ValueError):
        metrics.loss_group_ids({'question_family':'S2.13','catalog_case_id':'S2.13.999'})


@pytest.fixture(scope='module')
def real_tokenizer():
    """Load the frozen tokenizer without changing the Transformers torch feature flag."""
    from transformers import AutoTokenizer
    pin=json.loads((ROOT/'runtime/world_model/s2_production_v1_8k.json').read_text())['tokenizer']
    return AutoTokenizer.from_pretrained(ROOT/pin['path'],local_files_only=True)


def test_actual_mixed_rows_keep_production_masks_and_segment_every_assistant_token(real_tokenizer):
    manifest=training.load_corpus_manifest(ROOT/'data/world_model_v4/corpora/s2_production_v1_8k/manifest.json')
    selected={}
    expected=training.S2_FAMILIES | training.V3_TRACE_FAMILIES
    with (ROOT/'data/world_model_v4/corpora/s2_production_v1_8k/train.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line)
            selected.setdefault(row['metadata']['question_family'],row)
            if set(selected)==expected: break
    assert set(selected)==expected
    for family,row in selected.items():
        encoded=training.tokenize_training_row(row,real_tokenizer,expected_split='train',
            contract_routes=manifest['contract_routes'])
        plain=training.tokenize_row(row,real_tokenizer,expected_split='train',contract_routes=manifest['contract_routes'])
        assert encoded['input_ids']==plain['input_ids']
        assert encoded['completion_mask']==plain['completion_mask']
        assert encoded['labels']==plain['labels']
        assert all(bool(segment)==bool(active) for segment,active in zip(encoded['loss_segments'],encoded['completion_mask']))
        assert len(encoded['input_ids'])<=8192
        assert metrics.loss_group_ids(row['metadata'])


@pytest.fixture(scope='module')
def trainer():
    spec=importlib.util.spec_from_file_location('s2_trainer_test',ROOT/'scripts/train_sft_s2_dp_tp_ep.py')
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_periodic_validation_and_final_validation_have_distinct_steps(trainer,monkeypatch,tmp_path):
    """Exercise the actual loop on CPU to catch validation/control-flow regressions."""
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(1.0))
            self._device_mesh=SimpleNamespace(get_group=lambda:None)
    model=Model()
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    events=[]
    monkeypatch.setattr(trainer,'move_batch',lambda batch:batch)
    monkeypatch.setattr(trainer,'assert_identical_tp_batch',lambda *a,**kw:[])
    monkeypatch.setattr(trainer,'average_lora_gradients',lambda *a:None)
    monkeypatch.setattr(trainer,'assert_first_step_gradient_contract',lambda *a:None)
    monkeypatch.setattr(trainer,'assert_tp_replicated_lora_parity',lambda *a,**kw:1)
    monkeypatch.setattr(trainer,'assert_finite_lora_parameters',lambda *a:None)
    monkeypatch.setattr(trainer,'assert_dp_lora_parameter_parity',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(trainer,'_optimizer_parameters',lambda opt:list(model.parameters()))
    monkeypatch.setattr(trainer,'compute_loss_objective',lambda *a:{'loss':model.weight.square(),'completion_loss':model.weight.square()})
    monkeypatch.setattr(trainer,'save_checkpoint',lambda m,o,s,state,*a,**kw:events.append(('save',state.global_step)))
    def evaluate(*a,**kw):
        events.append((kw['trigger'],kw['global_step']))
        return {'elapsed_seconds':0.0}
    monkeypatch.setattr(trainer,'evaluate_validation',evaluate)
    args=SimpleNamespace(num_train_epochs=1,seed=1,preserve_dataset_order=True,
        per_device_train_batch_size=1,gradient_accumulation_steps=1,
        autograd_multithreading=True,max_grad_norm=1.,logging_steps=100,
        loss_contract='multiplex_sft_v3',save_strategy='steps',save_steps=2,
        eval_steps=2,throughput_calibration_updates=100,segment_loss_telemetry=False)
    topology=SimpleNamespace(data_parallel_size=1,global_batch_size=1)
    monkeypatch.setattr(trainer,'asdict',lambda obj:{})
    row={'input_ids':torch.tensor([1]),'attention_mask':torch.tensor([1])}
    state=trainer.train_loop(model,[row]*5,[],lambda items:{k:torch.stack([i[k] for i in items]) for k in row},
        optimizer,scheduler,trainer.LoopState(),5,args,topology,tmp_path,0,0,None)
    assert state.global_step==5
    assert events==[('save',2),('steps',2),('save',4),('steps',4),('epoch_end',5)]


@pytest.mark.parametrize('preparation_workers',[1,2])
def test_real_representative_families_build_and_reload_arrow_cache(
        trainer,real_tokenizer,monkeypatch,tmp_path,preparation_workers):
    """Run the production scalar/Arrow/token pipeline over all mixed families."""
    if preparation_workers>1:
        import socket
        try:
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                probe.bind(str(tmp_path/'multiprocessing-probe.sock'))
                probe.listen(1)
        except PermissionError:
            pytest.skip('The sandbox disables local multiprocessing socket IPC')
    expected=training.S2_FAMILIES | training.V3_TRACE_FAMILIES
    selected={}
    with (ROOT/'data/world_model_v4/corpora/s2_production_v1_8k/train.jsonl').open() as handle:
        for line in handle:
            row=json.loads(line)
            selected.setdefault(row['metadata']['question_family'],row)
            if set(selected)==expected: break
    path=tmp_path/'representative_train.jsonl'
    path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in selected.values()))
    monkeypatch.setenv('S2_CORPUS_MANIFEST_PATH',str(ROOT/'data/world_model_v4/corpora/s2_production_v1_8k/manifest.json'))
    import datasets
    monkeypatch.setattr(datasets.config,'HF_DATASETS_CACHE',tmp_path/'arrow')
    options=dict(max_length=8192,label='training',tokenizer_identity_sha256='a'*64,
        loss_contract='multiplex_sft_v3',dataset_id=training.S2_DATASET_ID,segment_loss_telemetry=True)
    dataset,families,identities=trainer.load_tokenized_dataset(str(path),real_tokenizer,0,**options)
    reloaded,_,identities2=trainer.load_tokenized_dataset(str(path),real_tokenizer,0,**options)
    assert set(families)==expected
    assert len(dataset)==len(expected)
    assert dataset._fingerprint==reloaded._fingerprint
    assert identities['record_ids']==identities2['record_ids']
    assert dataset[0]==reloaded[0]
    assert identities['flattened_cache_files'] and dataset.cache_files
    parallel, parallel_families, parallel_identities = trainer.load_tokenized_dataset(
        str(path),real_tokenizer,0,preparation_workers=preparation_workers,**options)
    assert len(parallel.cache_files)==preparation_workers
    assert parallel._fingerprint==dataset._fingerprint
    assert parallel_families==families
    assert parallel_identities['record_ids']==identities['record_ids']
    for index in range(len(dataset)):
        assert parallel[index]==dataset[index]
    stats=trainer.arrow_length_statistics(dataset)
    assert stats['rows']==len(expected) and 0<stats['max_input_tokens']<=8192
    receipt=tmp_path/'prepared.json'
    receipt.write_text(json.dumps({'datasets':{'training':{
        'rows':len(dataset),'fingerprint':dataset._fingerprint,
        'flattened_fingerprint':identities['flattened_fingerprint'],
        'flattened_cache_files':identities['flattened_cache_files'],
        'tokenized_cache_files':[item['filename'] for item in parallel.cache_files]}}}))
    monkeypatch.setenv('S2_PREPARED_CACHE_REPORT',str(receipt))
    monkeypatch.setattr(trainer.dist,'is_initialized',lambda:True)
    def forbidden_rebuild(*args,**kwargs):
        raise AssertionError('A GPU cache load attempted CPU preprocessing')
    monkeypatch.setattr(trainer.Dataset,'from_generator',forbidden_rebuild)
    monkeypatch.setattr(trainer.Dataset,'map',forbidden_rebuild)
    prepared,_,prepared_identities=trainer.load_tokenized_dataset(str(path),real_tokenizer,0,**options)
    assert prepared[0]==dataset[0]
    assert prepared._fingerprint==dataset._fingerprint
    assert prepared_identities['record_ids']==identities['record_ids']
    for index in range(len(dataset)):
        assert prepared[index]==dataset[index]


def test_canonical_serializer_and_sealed_code_hash_both_invalidate_token_cache(trainer,monkeypatch):
    """Cached masks cannot survive a change to the wire renderer or sealed code."""
    serializer_pin=['b'*64]
    monkeypatch.setattr(trainer,'sha256_file',lambda path:serializer_pin[0]
        if Path(path).name=='s2_3_tokens.py' else 'a'*64)
    monkeypatch.setattr(trainer,'tokenizer_artifact_hashes',lambda path:{'tokenizer.json':'f'*64})
    monkeypatch.setenv('S2_CORPUS_MANIFEST_PATH','/fixture/manifest.json')
    monkeypatch.setenv('S0_TRAINING_CODE_SHA256','c'*64)
    args=SimpleNamespace(dataset_path='/fixture/train.jsonl',validation_dataset_path='/fixture/validation.jsonl',
        corpus_manifest_path='/fixture/manifest.json',tokenizer_path='/fixture/tokenizer',
        max_length=8192,segment_loss_telemetry=True)
    options=dict(source_fingerprint='source',max_length=8192,tokenizer_identity_sha256='f'*64,
        loss_contract='multiplex_sft_v3',segment_loss_telemetry=True)
    original_fingerprint=trainer._tokenization_cache_fingerprint(**options)
    original_identity=trainer.cache_input_identity(args)
    serializer_pin[0]='d'*64
    changed_fingerprint=trainer._tokenization_cache_fingerprint(**options)
    changed_identity=trainer.cache_input_identity(args)
    assert changed_fingerprint!=original_fingerprint
    assert changed_identity!=original_identity
    assert changed_identity['canonical_serializer']=='d'*64
    monkeypatch.setenv('S0_TRAINING_CODE_SHA256','e'*64)
    assert trainer._tokenization_cache_fingerprint(**options)!=changed_fingerprint
    assert trainer.cache_input_identity(args)!=changed_identity
    monkeypatch.delenv('S0_TRAINING_CODE_SHA256')
    assert trainer.cache_input_identity(args)['training_code_sha256'] is None


@pytest.mark.parametrize('workers,allocated,prepare,valid', [
    (8,8,True,True),(8,2,True,False),(0,8,True,False),(8,7,False,True)])
def test_preparation_workers_respect_cpu_allocation_without_affecting_training(
        trainer,monkeypatch,tmp_path,workers,allocated,prepare,valid):
    """Limit preprocessing processes to the CPU job, leaving GPU topology alone."""
    import sys
    validation=tmp_path/'validation.jsonl'
    validation.write_text('')
    argv=['train_sft_s2_dp_tp_ep.py','--model_path','model','--tokenizer_path','tokenizer',
        '--dataset_path','train.jsonl','--validation_dataset_path',str(validation),
        '--corpus_manifest_path','manifest.json','--output_dir',str(tmp_path),
        '--loss_contract','multiplex_sft_v3','--max_length','8192','--bf16']
    if prepare: argv.append('--prepare-data-only')
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setenv('S2_PREPARATION_WORKERS',str(workers))
    monkeypatch.setenv('SLURM_CPUS_PER_TASK',str(allocated))
    monkeypatch.setattr(trainer,'load_corpus_manifest',lambda _: {})
    args=trainer.parse_args()
    assert args.preparation_workers==workers
    if valid:
        trainer.validate_cli(args)
    else:
        with pytest.raises(ValueError,match='preparation'):
            trainer.validate_cli(args)
