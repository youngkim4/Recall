"""The packaged server ships each facts file that the code reads from beside itself."""
import ast
from pathlib import Path

import pytest

import meaning_index.model_facts
import recall_services.models

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "recall_server.spec"


def bundled() -> dict[Path, str]:
    """Each file or folder that the spec ships, with the folder of the bundle that it goes to."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BUNDLE_DATAS":
            rows = eval(compile(ast.Expression(node.value), str(SPEC), "eval"), {"ROOT": ROOT, "str": str})  # noqa: S307
            return {Path(source): target for source, target in rows}
    raise AssertionError("the spec names no BUNDLE_DATAS")


@pytest.mark.parametrize("facts", [recall_services.models.FACTS_PATH, meaning_index.model_facts.FACTS_PATH])
def test_a_facts_file_is_shipped_into_the_folder_of_its_package(facts):
    shipped = bundled()
    assert facts.is_file() and facts in shipped
    assert shipped[facts] == facts.parent.name
