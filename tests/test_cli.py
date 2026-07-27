import subprocess
import sys
from pathlib import Path

from swave.cli import main


def test_cli_import_does_not_eagerly_load_matplotlib() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import swave.cli; "
                "assert 'matplotlib.pyplot' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_help_lists_all_workflows(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in (
        "generate",
        "train",
        "evaluate",
        "predict",
        "plot-model",
        "plot-dispersion",
    ):
        assert command in output


def test_plot_model_creates_nonempty_png(tmp_path: Path) -> None:
    output = tmp_path / "model.png"
    assert (
        main(
            [
                "plot-model",
                "--sample-id",
                "4",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.stat().st_size > 1_000


def test_invalid_predict_input_returns_clean_error(
    tmp_path: Path, capsys
) -> None:
    code = main(["predict", str(tmp_path / "missing.pt"), "0.5"])
    assert code == 2
    assert capsys.readouterr().err.startswith("error:")
