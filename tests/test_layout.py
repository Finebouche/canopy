"""
Layout of canopy (CLAUDE.md, README): one folder per study in canopy/studies, each file of a study importing only the
shared modules (settings, common, datasets, models, studies.common) and the files of its own study; task.py imports
only the shared modules; the song generator builds on the next phrase and the generation of syllables, the
control next_phrase_labels on the next phrase, and the control next_syllable on the next phrase and on the segmentation
of the syllables of syllable_generation.
"""
import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "canopy"
SHARED = ("canopy.settings", "canopy.common", "canopy.datasets", "canopy.models", "canopy.studies.common")
BUILDS_ON = {"song_generation": ("next_phrase", "syllable_generation"), "next_phrase_labels": ("next_phrase",),
             "next_syllable": ("next_phrase", "syllable_generation")}


def imports(path):
    tree = ast.parse(path.read_text())
    return ({node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
             and node.module.startswith("canopy")}
            | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
               if alias.name.startswith("canopy")})


def test_each_study_imports_only_the_shared_modules_and_its_own_files():
    for path in sorted((PACKAGE / "studies").rglob("*.py")):
        if path.parent == PACKAGE / "studies":
            continue  # (common.py, __init__.py)
        study = path.parent.name
        allowed = SHARED + tuple(f"canopy.studies.{s}" for s in (study, *BUILDS_ON.get(study, ())))
        if path.stem == "task":  # (a task imports only the shared modules, and the tasks its study builds on)
            allowed = SHARED + tuple(f"canopy.studies.{s}.task" for s in BUILDS_ON.get(study, ()))
        for module in imports(path):
            assert module.startswith(allowed), f"{study}/{path.name} imports {module}"


def test_the_shared_modules_import_no_study():
    shared = [PACKAGE / "settings.py", PACKAGE / "common.py", PACKAGE / "studies" / "common.py",
              *(PACKAGE / "datasets").rglob("*.py"), *(PACKAGE / "models").rglob("*.py")]
    for path in shared:
        for module in imports(path):
            assert module.startswith(SHARED), f"{path.relative_to(PACKAGE)} imports {module}"


def test_the_studies_of_the_reservoirs_are_named_study():
    for study in ("phrase_boundaries", "next_phrase", "syllables_classification"):
        assert {"task.py", "study.py", "evaluation.py"} <= {p.name for p in (PACKAGE / "studies" / study).glob("*.py")}
    assert not [p for p in PACKAGE.rglob("*.py") if "hpo" in p.stem]
