import importlib.util
import json
import zipfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "import_results", Path(__file__).resolve().parents[1] / "scripts" / "import_results.py")
import_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_results)


def _zip(tmp_path, entries):
    z = tmp_path / "run.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return z


def test_import_ok_and_no_overwrite(tmp_path):
    z = _zip(tmp_path, {"meta.json": json.dumps({"phase": "p0", "run_id": "r1"}), "hello.json": "{}"})
    root = tmp_path / "results"
    assert import_results.main([str(z), "--results-root", str(root)]) == 0
    assert (root / "p0" / "r1" / "hello.json").exists()
    assert import_results.main([str(z), "--results-root", str(root)]) == 1          # exists
    assert import_results.main([str(z), "--results-root", str(root), "--force"]) == 0


def test_import_rejects_zip_slip(tmp_path):
    z = _zip(tmp_path, {"meta.json": json.dumps({"phase": "p0", "run_id": "r2"}), "../../evil.txt": "x"})
    root = tmp_path / "results"
    assert import_results.main([str(z), "--results-root", str(root)]) == 1
    assert not (tmp_path / "evil.txt").exists()
    assert not (root / "p0" / "r2").exists()


def test_import_requires_meta(tmp_path):
    z = _zip(tmp_path, {"hello.json": "{}"})
    assert import_results.main([str(z), "--results-root", str(tmp_path / "r")]) == 1
