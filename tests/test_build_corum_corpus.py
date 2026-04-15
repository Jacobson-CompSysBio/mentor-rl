import csv
import json
from pathlib import Path

from scripts import build_corum_corpus as bcc


def _write_corum_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "complex_id",
        "complex_name",
        "synonyms",
        "organism",
        "cell_line",
        "pmid",
        "comment_complex",
        "comment_members",
        "comment_disease",
        "comment_drug",
        "comment_drug_formal",
        "subunits_uniprot_id",
        "subunits_gene_name",
        "subunits_gene_name_synonyms",
        "functions_evi",
        "functions_pmid",
        "functions_go_id",
        "functions_go_name",
        "fcgs_id",
        "fcgs_name",
        "fcgs_category_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_simple_flist(tmp_path: Path, multiplex_gene_ids: list[str]) -> Path:
    network_path = tmp_path / "layer1.tsv"
    flist_path = tmp_path / "multiplex.flist"
    with network_path.open("w", encoding="utf-8") as handle:
        for left, right in zip(multiplex_gene_ids[:-1], multiplex_gene_ids[1:]):
            handle.write(f"{left}\t{right}\t1.0\n")
    flist_path.write_text(f"{network_path}\tlayer1\t1\n", encoding="utf-8")
    return flist_path


def _fake_hit(symbol: str, ensembl_gene_id: str, *, alias=None, uniprot=None) -> dict:
    payload = {
        "_id": ensembl_gene_id.replace("ENSG", ""),
        "symbol": symbol,
        "ensembl": {"gene": ensembl_gene_id},
        "taxid": 9606,
    }
    if alias is not None:
        payload["alias"] = alias
    if uniprot is not None:
        payload["uniprot"] = {"Swiss-Prot": uniprot}
    return payload


class FakeResolver:
    MEMBER_MAP = {
        "GENEA": ("ENSG000001", "GENEA", "symbol"),
        "GENEB": ("ENSG000002", "GENEB", "uniprot"),
        "GENEC": ("ENSG000003", "GENEC", "alias"),
        "GENED": ("ENSG000004", "GENED", "symbol"),
        "GENEE": ("ENSG000005", "GENEE", "symbol"),
        "GENEF": ("ENSG000006", "GENEF", "symbol"),
        "GENEG": ("ENSG000007", "GENEG", "symbol"),
        "GENEH": ("ENSG000008", "GENEH", "symbol"),
        "GENEI": ("ENSG000009", "GENEI", "symbol"),
        "GENEJ": ("ENSG000010", "GENEJ", "symbol"),
        "GENEK": ("ENSG000011", "GENEK", "symbol"),
        "GENEL": ("ENSG000012", "GENEL", "symbol"),
        "GENEM": ("ENSG000013", "GENEM", "symbol"),
        "GENEN": ("ENSG000014", "GENEN", "symbol"),
        "GENEO": ("ENSG000020", "GENEO", "symbol"),
        "GENEP": ("ENSG000021", "GENEP", "symbol"),
    }

    REVERSE_SYMBOLS = {
        "ENSG000015": "NOISE1",
        "ENSG000016": "NOISE2",
        "ENSG000017": "NOISE3",
        "ENSG000018": "NOISE4",
        "ENSG000019": "NOISE5",
    }

    def __init__(self, cache_dir, multiplex_genes=None, batch_size=500):
        self.multiplex_genes = multiplex_genes or set()

    def prefetch(self, scope, queries):
        return None

    def resolve_member(self, gene_symbol, uniprot_id, aliases):
        if gene_symbol not in self.MEMBER_MAP:
            return {
                "status": "unresolved",
                "ensembl_gene_id": None,
                "display_symbol": gene_symbol,
                "resolved_via": None,
                "matched_query": None,
                "hit_ids": [],
                "candidate_count": 0,
            }
        gene_id, display_symbol, resolved_via = self.MEMBER_MAP[gene_symbol]
        assert gene_id in self.multiplex_genes
        return {
            "status": "resolved",
            "ensembl_gene_id": gene_id,
            "display_symbol": display_symbol,
            "resolved_via": resolved_via,
            "matched_query": gene_symbol,
            "hit_ids": [gene_id],
            "candidate_count": 1,
        }

    def get_gene_annotation(self, ensembl_gene_id):
        return {
            "ensembl_gene_id": ensembl_gene_id,
            "symbol": self.REVERSE_SYMBOLS.get(ensembl_gene_id, ensembl_gene_id),
            "resolved_via": "ensembl.gene",
            "hit_ids": [ensembl_gene_id],
        }


def test_parse_corum_row_aligns_members_and_go_pairs():
    row = {
        "complex_id": "1",
        "complex_name": "Alpha complex",
        "synonyms": "AltA;AltB",
        "organism": "Human",
        "cell_line": "HeLa",
        "pmid": "12345",
        "comment_complex": "Alpha note",
        "comment_members": "",
        "comment_disease": "",
        "comment_drug": "",
        "comment_drug_formal": "",
        "subunits_uniprot_id": "UPA;UPB;UPC",
        "subunits_gene_name": "GENEA;GENEB;GENEC",
        "subunits_gene_name_synonyms": "ALIA1,ALIA2;ALIB1;ALIC1,ALIC2",
        "functions_evi": "",
        "functions_pmid": "",
        "functions_go_id": "GO:0001;GO:0002",
        "functions_go_name": "first mechanism;second mechanism",
        "fcgs_id": "11",
        "fcgs_name": "Transport",
        "fcgs_category_name": "CategoryA",
    }

    parsed = bcc.parse_corum_row(row)

    assert parsed["source_complex_id"] == 1
    assert parsed["members_raw"][0]["source_symbol"] == "GENEA"
    assert parsed["members_raw"][0]["source_aliases"] == ["ALIA1", "ALIA2"]
    assert parsed["members_raw"][1]["source_aliases"] == ["ALIB1"]
    assert parsed["go_terms"] == [
        {"go_id": "GO:0001", "go_name": "first mechanism"},
        {"go_id": "GO:0002", "go_name": "second mechanism"},
    ]


def test_mygene_resolver_prefers_symbol_then_uniprot_then_alias(tmp_path):
    resolver = bcc.MyGeneResolver(cache_dir=tmp_path, multiplex_genes={"ENSG000001", "ENSG000002", "ENSG000003"})
    resolver.cache[resolver.cache_key("symbol", "GENEA")] = [
        _fake_hit("GENEA", "ENSG000001", alias=["ALIAA"], uniprot="UPA")
    ]
    resolver.cache[resolver.cache_key("symbol", "GENEB")] = []
    resolver.cache[resolver.cache_key("uniprot", "UPB")] = [
        _fake_hit("GENEB", "ENSG000002", alias=["ALIB"], uniprot="UPB")
    ]
    resolver.cache[resolver.cache_key("symbol", "GENEC")] = []
    resolver.cache[resolver.cache_key("uniprot", "UPC")] = []
    resolver.cache[resolver.cache_key("alias", "ALIC")] = [
        _fake_hit("GENEC", "ENSG000003", alias=["ALIC"], uniprot="UPC")
    ]
    resolver.cache[resolver.cache_key("symbol", "AMBIG")] = [
        _fake_hit("AMBIG", "ENSG000001", alias=["AMBIG"], uniprot="U1"),
        _fake_hit("AMBIG", "ENSG000002", alias=["AMBIG"], uniprot="U2"),
    ]

    symbol_resolution = resolver.resolve_member("GENEA", "UPA", ["ALIAA"])
    uniprot_resolution = resolver.resolve_member("GENEB", "UPB", ["ALIB"])
    alias_resolution = resolver.resolve_member("GENEC", "UPC", ["ALIC"])
    ambiguous_resolution = resolver.resolve_member("AMBIG", "U1", ["AMBIG"])

    assert symbol_resolution["resolved_via"] == "symbol"
    assert symbol_resolution["ensembl_gene_id"] == "ENSG000001"
    assert uniprot_resolution["resolved_via"] == "uniprot"
    assert uniprot_resolution["ensembl_gene_id"] == "ENSG000002"
    assert alias_resolution["resolved_via"] == "alias"
    assert alias_resolution["ensembl_gene_id"] == "ENSG000003"
    assert ambiguous_resolution["status"] == "unresolved"


def test_assign_splits_has_no_duplicate_membership():
    complexes = []
    for index in range(10):
        complexes.append(
            {
                "complex_record_id": f"corum_complex_{index:05d}",
                "size_bin": "3",
                "has_fcgs": True,
            }
        )

    assigned = bcc.assign_splits(complexes, seed=7)

    assert len(assigned) == 10
    assert len({row["complex_record_id"] for row in assigned}) == 10
    counts = {split: sum(row["split"] == split for row in assigned) for split in bcc.SPLITS}
    assert counts == {"train": 8, "val": 1, "test": 1}


def test_build_corum_corpus_smoke_and_determinism(tmp_path, monkeypatch):
    corum_path = tmp_path / "corum.tsv"
    flist_path = _write_simple_flist(
        tmp_path,
        [
            "ENSG000001",
            "ENSG000002",
            "ENSG000003",
            "ENSG000004",
            "ENSG000005",
            "ENSG000006",
            "ENSG000007",
            "ENSG000008",
            "ENSG000009",
            "ENSG000010",
            "ENSG000011",
            "ENSG000012",
            "ENSG000013",
            "ENSG000014",
            "ENSG000015",
            "ENSG000016",
            "ENSG000017",
            "ENSG000018",
            "ENSG000019",
            "ENSG000020",
            "ENSG000021",
        ],
    )
    rows = [
        {
            "complex_id": "1",
            "complex_name": "Alpha complex",
            "synonyms": "AlphaA",
            "organism": "Human",
            "cell_line": "HeLa",
            "pmid": "1001",
            "comment_complex": "Alpha mechanistic note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UA;UB;UC",
            "subunits_gene_name": "GENEA;GENEB;GENEC",
            "subunits_gene_name_synonyms": "ALIAA;ALIB;ALIC",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0001;GO:0002",
            "functions_go_name": "alpha process;alpha location",
            "fcgs_id": "11",
            "fcgs_name": "Transport",
            "fcgs_category_name": "CategoryA",
        },
        {
            "complex_id": "2",
            "complex_name": "Alpha duplicate",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "HeLa",
            "pmid": "1002",
            "comment_complex": "Duplicate alpha note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UA;UB;UC",
            "subunits_gene_name": "GENEA;GENEB;GENEC",
            "subunits_gene_name_synonyms": "ALIAA;ALIB;ALIC",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0001;GO:0002",
            "functions_go_name": "alpha process;alpha location",
            "fcgs_id": "11",
            "fcgs_name": "Transport",
            "fcgs_category_name": "CategoryA",
        },
        {
            "complex_id": "3",
            "complex_name": "Beta complex",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "293",
            "pmid": "1003",
            "comment_complex": "Beta note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UD;UE",
            "subunits_gene_name": "GENED;GENEE",
            "subunits_gene_name_synonyms": "ALID;ALIE",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0003",
            "functions_go_name": "beta process",
            "fcgs_id": "",
            "fcgs_name": "",
            "fcgs_category_name": "",
        },
        {
            "complex_id": "4",
            "complex_name": "Gamma complex",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "293",
            "pmid": "1004",
            "comment_complex": "Gamma note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UF;UG;UH;UI",
            "subunits_gene_name": "GENEF;GENEG;GENEH;GENEI",
            "subunits_gene_name_synonyms": "ALIF;ALIG;ALIH;ALII",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0004;GO:0005",
            "functions_go_name": "gamma process;gamma location",
            "fcgs_id": "",
            "fcgs_name": "",
            "fcgs_category_name": "",
        },
        {
            "complex_id": "5",
            "complex_name": "Delta complex",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "293",
            "pmid": "1005",
            "comment_complex": "Delta note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UJ;UK",
            "subunits_gene_name": "GENEJ;GENEK",
            "subunits_gene_name_synonyms": "ALIJ;ALIK",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0006",
            "functions_go_name": "delta process",
            "fcgs_id": "22",
            "fcgs_name": "Signaling",
            "fcgs_category_name": "CategoryB",
        },
        {
            "complex_id": "6",
            "complex_name": "Epsilon complex",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "293",
            "pmid": "1006",
            "comment_complex": "Epsilon note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UL;UM;UN",
            "subunits_gene_name": "GENEL;GENEM;GENEN",
            "subunits_gene_name_synonyms": "ALIL;ALIM;ALIN",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0007",
            "functions_go_name": "epsilon process",
            "fcgs_id": "",
            "fcgs_name": "",
            "fcgs_category_name": "",
        },
        {
            "complex_id": "7",
            "complex_name": "Zeta complex",
            "synonyms": "",
            "organism": "Human",
            "cell_line": "293",
            "pmid": "1007",
            "comment_complex": "Zeta note",
            "comment_members": "",
            "comment_disease": "",
            "comment_drug": "",
            "comment_drug_formal": "",
            "subunits_uniprot_id": "UO;UP",
            "subunits_gene_name": "GENEO;GENEP",
            "subunits_gene_name_synonyms": "ALIO;ALIP",
            "functions_evi": "",
            "functions_pmid": "",
            "functions_go_id": "GO:0008",
            "functions_go_name": "zeta process",
            "fcgs_id": "",
            "fcgs_name": "",
            "fcgs_category_name": "",
        },
    ]
    _write_corum_tsv(corum_path, rows)

    monkeypatch.setattr(bcc, "MyGeneResolver", FakeResolver)

    out_dir_1 = tmp_path / "out1"
    cache_dir_1 = tmp_path / "cache1"
    result_1 = bcc.build_corum_corpus(
        corum_path=corum_path,
        multiplex_flist=flist_path,
        out_dir=out_dir_1,
        seed=11,
        min_complex_size=2,
        cache_dir=cache_dir_1,
    )
    out_dir_2 = tmp_path / "out2"
    cache_dir_2 = tmp_path / "cache2"
    result_2 = bcc.build_corum_corpus(
        corum_path=corum_path,
        multiplex_flist=flist_path,
        out_dir=out_dir_2,
        seed=11,
        min_complex_size=2,
        cache_dir=cache_dir_2,
    )

    assert result_1["complex_rows"] == result_2["complex_rows"]
    assert result_1["tasks"] == result_2["tasks"]

    required_files = [
        "manifest.json",
        "complexes.jsonl",
        "tasks.train.jsonl",
        "tasks.val.jsonl",
        "tasks.test.jsonl",
        "split_report.json",
    ]
    for file_name in required_files:
        assert (out_dir_1 / file_name).exists()

    assert result_1["manifest"]["complex_count"] == 6
    sample_task = result_1["tasks"][0]
    assert {
        "task_id",
        "split",
        "task_type",
        "difficulty",
        "query_text",
        "query_template_id",
        "evidence_mode",
        "visible_inputs",
        "hidden_target",
        "mechanism_labels",
        "normalization",
        "provenance",
    } <= set(sample_task.keys())

    complex_gene_sets = {
        row["complex_record_id"]: set(row["gene_ids"]) for row in result_1["complex_rows"]
    }
    gene_to_complexes = {}
    for complex_record_id, gene_ids in complex_gene_sets.items():
        for gene_id in gene_ids:
            gene_to_complexes.setdefault(gene_id, set()).add(complex_record_id)

    for task in result_1["tasks"]:
        seed_gene_ids = set(task["visible_inputs"]["seed_gene_ids"])
        if task["task_type"] == "explanation":
            assert seed_gene_ids == set(task["hidden_target"]["target_gene_ids"])
        elif task["task_type"] == "recovery":
            target_gene_ids = set(task["hidden_target"]["target_gene_ids"])
            assert seed_gene_ids < target_gene_ids
            assert len(seed_gene_ids) >= 2
        elif task["task_type"] == "refinement":
            target_gene_ids = set(task["hidden_target"]["target_gene_ids"])
            noise_gene_ids = seed_gene_ids - target_gene_ids
            assert target_gene_ids <= seed_gene_ids
            assert noise_gene_ids
            target_complex_ids = set()
            for gene_id in target_gene_ids:
                target_complex_ids.update(gene_to_complexes.get(gene_id, set()))
            for noise_gene_id in noise_gene_ids:
                assert not (gene_to_complexes.get(noise_gene_id, set()) & target_complex_ids)
        else:
            assert task["hidden_target"]["target_gene_ids"] is None
            for gene_ids in complex_gene_sets.values():
                assert len(seed_gene_ids & gene_ids) <= 1

    split_report = json.loads((out_dir_1 / "split_report.json").read_text(encoding="utf-8"))
    assert "cross_split_shared_genes" in split_report
