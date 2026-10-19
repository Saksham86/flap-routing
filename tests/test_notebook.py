"""Every code cell of the Colab runner must be valid Python (shell '!' and '%' magic lines aside)."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "run_colab.ipynb"


def test_notebook_cells_parse():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    errors = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        lines = [(l[:len(l) - len(l.lstrip())] + "pass") if l.lstrip().startswith(("!", "%")) else l
                 for l in "".join(c["source"]).splitlines()]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as e:
            errors.append(f"cell {i}: {e}")
    assert not errors, errors
