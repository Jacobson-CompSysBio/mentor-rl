"""Check that every S2 generator entrypoint can load its command-line parser."""

import importlib.util
from pathlib import Path
import runpy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = ["build_world_model_s2.py", *[
    f"build_world_model_s2_{family}{'_v2' if family == 11 else ''}.py"
    for family in range(2, 14)
]]


@pytest.mark.parametrize("builder", BUILDERS)
def test_generator_help_loads_without_building(builder, monkeypatch, capsys):
    """Catch missing imports at the public entrypoint before any source work."""
    monkeypatch.setattr(sys, "argv", [builder, "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(ROOT / "scripts" / builder), run_name="__main__")
    assert stopped.value.code == 0
    assert "--output-dir" in capsys.readouterr().out


@pytest.mark.parametrize("builder", BUILDERS)
def test_generator_cli_forwards_its_verification_schedule(builder, monkeypatch, tmp_path, capsys):
    """Check the executable smoke path calls its builder with the selected budget."""
    spec = importlib.util.spec_from_file_location("s2_generator_cli", ROOT / "scripts" / builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []

    def build(output_dir, **kwargs):
        called.append((output_dir, kwargs))
        return {"purpose": "builder_verification", "generation_budget": None,
                "row_counts": {}, "panel_id": "verification", "training_ready": False,
                "production_export_allowed": False}

    monkeypatch.setattr(module, "build_corpus", build)
    first_family = builder == "build_world_model_s2.py"
    argv = [builder, "--output-dir", str(tmp_path)] + ([] if first_family else ["--smoke"])
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(called) == 1
    assert called[0][0] == tmp_path
    if first_family:
        assert called[0][1]["counts"] == module.DEFAULT_COUNTS
    else:
        assert called[0][1]["smoke"] is True
        assert called[0][1]["counts"] is None
    assert '"training_ready": false' in capsys.readouterr().out
