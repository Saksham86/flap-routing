# flap-routing (QACR)

Query-adaptive compression-level routing for edge LLM inference (B.Tech thesis).

FLAP prunes one small LLM (SmolLM2-1.7B) into a nested ladder of compression levels. A cheap router picks the level per query, and the system is evaluated on accuracy vs latency/energy/memory, on a Colab T4 and on an edge device.

- Plan and rules: [`CLAUDE.md`](CLAUDE.md)
- Progress: [`STATUS.md`](STATUS.md)
- Related work: [`thesis/related_work_notes.md`](thesis/related_work_notes.md)

## Quick start (local, CPU)
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use .venv/bin/activate on Linux/macOS
pip install torch                                    # CPU build is fine locally
pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
python scripts/p0_hello_gpu.py --config configs/p0_hello.yaml
```

## GPU runs
Open `notebooks/run_colab.ipynb` in Colab (T4 runtime), set `REPO_URL` and the script/config, and run all cells. Results go to `MyDrive/qacr/results/<phase>/<run_id>/`. Then import the downloaded zip:
```bash
python scripts/import_results.py <run_id>.zip
```
