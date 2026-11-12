"""Project the mixed S2 production corpus onto trainer and evaluator interfaces.

Family routes bind each S2 row to its manifest-pinned prompt, tools, and overlay.
Historical replay keeps its explicit legacy instruction and original evidence.
The canonical complete-trace serializer supplies current-turn token supervision.
"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re

from ..world_model_training import V3_TRACE_FAMILIES, _validate_v3_message_sequence
from .s2_3_tokens import tokenize_trace
from .s2_corpus import SCHEMA_VERSION
from .s2_production_replay import LEGACY_INSTRUCTION, validate_replay_adapter


MAX_LENGTH = 8192
FAMILY_PATTERN = re.compile(r'S2\.(?:[1-9]|1[0-3])\Z')


def _json(value, *, sort_keys=False):
    """Keep finite compact JSON, preserving label-significant object order."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=sort_keys, allow_nan=False)


def tools_sha256(tools):
    """Hash compact ordered tool JSON using the production manifest convention."""
    return hashlib.sha256(_json(tools).encode()).hexdigest()


def _wire(row):
    """Require canonical context and a complete, reasoned current assistant turn."""
    if not isinstance(row, Mapping) or list(row) != ['row_id', 'messages', 'tools', 'metadata']:
        raise ValueError('the production row field order changed')
    if not isinstance(row['row_id'], str) or not row['row_id'].strip():
        raise ValueError('a production row requires a nonempty row_id')
    _validate_v3_message_sequence(row['messages'])
    meta = row['metadata']
    legacy = isinstance(meta, Mapping) and meta.get('source_stage') in ('S0', 'S1')
    if not legacy and sum(message['role'] == 'user' for message in row['messages']) != 1:
        raise ValueError('production rows require exactly one user turn')
    current_user = max(index for index, message in enumerate(row['messages']) if message['role'] == 'user')
    for index, message in enumerate(row['messages']):
        role = message['role']
        keys = ({'role', 'thinking', 'tool_call'} if 'tool_call' in message else
                {'role', 'thinking', 'content'} if role == 'assistant' else
                {'role', 'name', 'content'} if role == 'tool' else {'role', 'content'})
        prior_assistant = legacy and role == 'assistant' and index < current_user
        # Legacy context may omit analysis that the current-turn serializer removes.
        if prior_assistant and 'thinking' not in message:
            keys.remove('thinking')
        if set(message) != keys or role == 'assistant' and not prior_assistant and not message.get('thinking', '').strip():
            raise ValueError('unexpected message fields or missing current-turn reasoning')
    tools = row['tools']
    if not isinstance(tools, list) or not tools or any(
        not isinstance(tool, Mapping) or tool.get('type') != 'function'
        or not isinstance(tool.get('function'), Mapping)
        or not isinstance(tool['function'].get('name'), str)
        or not isinstance(tool['function'].get('description'), str) for tool in tools
    ):
        raise ValueError('production rows require described function tool definitions')
    names = [tool['function']['name'] for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError('production tool names must be unique')
    for index, message in enumerate(row['messages']):
        if 'tool_call' in message:
            name = message['tool_call']['name']
            if name not in names or row['messages'][index + 1]['name'] != name:
                raise ValueError('a production tool call and response require the declared same tool')
    if (not isinstance(meta, Mapping) or not isinstance(meta.get('split_unit'), str) or not meta['split_unit']
            or not isinstance(meta.get('question_family'), str) or not meta['question_family']):
        raise ValueError('production metadata requires its original nonempty split_unit')
    final = row['messages'][-1]['content']
    if (not isinstance(final['status'], str) or final['continuation_decision'] not in ('stop', 'continue', 'defer')
            or final['blocked_reason'] is not None and not isinstance(final['blocked_reason'], str)
            or (final['continuation_decision'] == 'defer') != bool(final['blocked_reason'])
            or not isinstance(final['summary'], str) or not final['summary'].strip()):
        raise ValueError('the final envelope has an invalid decision, blocker, or summary')
    return meta


def _validate_row(row, *, expected_split, contract_routes):
    """Route a row through exactly its declared source-stage contract."""
    if expected_split not in ('train', 'validation', 'test'):
        raise ValueError('unknown production split')
    meta = _wire(row)
    stage = meta.get('source_stage')
    if stage is not None:
        if stage not in ('S0', 'S1') or expected_split != 'train' or meta.get('split', 'train') != 'train':
            raise ValueError('legacy replay belongs only to the training split')
        validate_replay_adapter(row, stage)
        receipt = meta['production_replay']
        original_prompt = row['messages'][0]['content'][len(LEGACY_INSTRUCTION):]
        if (meta['question_family'] not in V3_TRACE_FAMILIES
                or hashlib.sha256(original_prompt.encode()).hexdigest() != receipt.get('source_system_prompt_sha256')
                or receipt.get('source_tool_contract_version') != meta.get('tool_contract_version')
                or receipt.get('source_trace') != meta.get('source_trace')):
            raise ValueError('legacy replay changed its original contract identity')
        return meta
    family, case_id = meta.get('question_family'), meta.get('catalog_case_id')
    if (not isinstance(family, str) or FAMILY_PATTERN.fullmatch(family) is None
            or not isinstance(case_id, str) or re.fullmatch(re.escape(family) + r'\.[1-9][0-9]*', case_id) is None):
        raise ValueError('the S2 family and catalog case identity do not match')
    if (meta.get('split') != expected_split or meta.get('schema_version') != SCHEMA_VERSION
            or meta.get('supervision') != 'current_turn_assistant_only'):
        raise ValueError('the S2 source split, schema, or supervision changed')
    route = contract_routes.get(family) if isinstance(contract_routes, Mapping) else None
    if not isinstance(route, Mapping) or set(route) != {'contract_version', 'system_prompt_sha256', 'tools_sha256'}:
        raise ValueError('the S2 family has no exact production contract route')
    prompt_hash = hashlib.sha256(row['messages'][0]['content'].encode()).hexdigest()
    if (not isinstance(route['contract_version'], str) or not route['contract_version']
            or meta.get('tool_contract_version') != route['contract_version']
            or prompt_hash != route['system_prompt_sha256'] or prompt_hash != meta.get('system_prompt_sha256')
            or tools_sha256(row['tools']) != route['tools_sha256']):
        raise ValueError('the S2 row differs from its pinned family contract')
    return meta


def flatten_row(row, *, expected_split, contract_routes):
    """Return the existing seven-scalar Arrow schema without changing the trace."""
    meta = _validate_row(row, expected_split=expected_split, contract_routes=contract_routes)
    return {'trace_json': _json(dict(row)), 'metadata_json': _json(dict(meta), sort_keys=True),
        'question_family': meta['question_family'], 'record_id': row['row_id'],
        'fact_id': meta['split_unit'], 'prompt_form_id': expected_split, 'split': expected_split}


def tokenize_row(row, tokenizer, *, expected_split, contract_routes):
    """Keep the complete wire and supervise every current-turn assistant segment."""
    _validate_row(row, expected_split=expected_split, contract_routes=contract_routes)
    encoded = tokenize_trace(row, tokenizer)
    ids, mask = encoded['input_ids'], encoded['completion_mask']
    if (not ids or len(ids) > MAX_LENGTH or len(mask) != len(ids) or not any(mask)
            or any(type(flag) is not int or flag not in (0, 1) for flag in mask)
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError('invalid complete production trace length or assistant supervision')
    return {'input_ids': list(ids), 'attention_mask': [1] * len(ids),
        'labels': [token if flag else -100 for token, flag in zip(ids, mask)], 'completion_mask': list(mask)}


def evaluation_target(row):
    """Extract structured targets and current-turn calls, leaving free prose free."""
    _wire(row)
    final = row['messages'][-1]['content']
    current_user = max(index for index, message in enumerate(row['messages']) if message['role'] == 'user')
    current = row['messages'][current_user + 1:]
    return deepcopy({field: final[field] for field in ('status', 'result', 'continuation_decision', 'provenance')} | {
        'blocked_reason_required': final['continuation_decision'] == 'defer',
        'expected_tool_calls': [message['tool_call'] for message in current if 'tool_call' in message],
        'expected_tool_responses': [{'name': message['name'], 'content': message['content']}
                                   for message in current if message['role'] == 'tool']})
