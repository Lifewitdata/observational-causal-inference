"""The notebooks are deliverables, so they are checked like one.

Committing an `.ipynb` whose outputs were never produced is worse than committing nothing: it looks
like evidence and is a stub. These tests assert the four notebooks on disk are valid, were actually
run, contain their figures, and carry no error trace.
"""

from pathlib import Path
import re

import nbformat
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"

EXPECTED = [
    "01_design.ipynb",
    "02_did_naive_vs_staggered.ipynb",
    "03_synthetic_control_prop99.ipynb",
    "04_sensitivity_and_writeup.ipynb",
]


@pytest.mark.parametrize("name", EXPECTED)
def test_notebook_is_valid_and_fully_executed(name):
    nb = nbformat.read(NOTEBOOKS / name, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, f"{name} has no code cells"
    assert all(isinstance(c.execution_count, int) for c in code), f"{name} was not executed"
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"{name} has {len(errors)} error output(s)"


def test_figures_are_embedded_in_the_notebooks():
    """The analysis notebooks draw their charts inline rather than only writing PNGs to disk."""
    for name in EXPECTED[:3]:
        nb = nbformat.read(NOTEBOOKS / name, as_version=4)
        images = [
            o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
            if "image/png" in o.get("data", {})
        ]
        assert images, f"{name} embeds no figure"


def test_notebooks_are_portable():
    """No absolute paths from the machine that built them: each notebook resolves the repo from its
    own working directory, so `make notebooks` works after a clone."""
    for name in EXPECTED:
        nb = nbformat.read(NOTEBOOKS / name, as_version=4)
        head = next(c.source for c in nb.cells if c.cell_type == "code")
        assert 'REPO = Path.cwd()' in head, f"{name} does not resolve paths relative to the repo"
        for c in nb.cells:
            assert not re.search(r"[A-Za-z]:\\|/Users/|/home/", c.source), (
                f"{name} hardcodes a machine-specific path"
            )


def test_report_figures_exist_on_disk():
    """The report references these by name, so a missing file is a broken deliverable."""
    figures = Path(__file__).resolve().parents[1] / "report" / "figures"
    names = [
        "fig1_event_study.png",
        "fig2_gantt.png",
        "fig3_sc.png",
        "fig4_gap.png",
        "fig5_placebo.png",
        "fig6_pairs.png",
    ]
    for n in names:
        p = figures / n
        assert p.exists(), f"{n} is missing"
        assert p.stat().st_size > 5_000, f"{n} looks like a stub"
