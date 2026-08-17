import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from swave.cli import main
from swave.config import (
    DatasetConfig,
    HybridInversionConfig,
    InversionConfig,
    PhysicsConfig,
)
from swave.inversion_results import ResultManifest
from swave.splits import SPLIT_POLICY


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
        "audit-dataset",
        "generate",
        "train",
        "train-inverse",
        "evaluate",
        "predict",
        "plot-model",
        "plot-dispersion",
        "invert",
        "inversion-report",
        "hybrid-invert",
        "hybrid-report",
    ):
        assert command in output


def test_installed_entrypoint_lists_inversion_workflows() -> None:
    executable = Path(sys.executable).with_name("swave")
    completed = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "invert" in completed.stdout
    assert "inversion-report" in completed.stdout


def test_cli_import_does_not_eagerly_load_inversion_workflows() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import swave.cli; "
                "assert 'swave.inversion_runner' not in sys.modules; "
                "assert 'swave.inversion_report' not in sys.modules; "
                "assert 'swave.hybrid_runner' not in sys.modules; "
                "assert 'swave.hybrid_report' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_invert_help_exposes_all_validated_overrides(capsys) -> None:
    assert main(["invert", "--help"]) == 0
    output = capsys.readouterr().out
    for option in (
        "--config",
        "--experiment",
        "--dataset-config",
        "--dataset-dir",
        "--checkpoint",
        "--output-dir",
        "--device",
        "--workers",
        "--deep-samples-per-job",
        "--threads-per-worker",
        "--task-index",
        "--task-count",
    ):
        assert option in output

    assert main(["inversion-report", "--help"]) == 0
    assert "--dataset-config" in capsys.readouterr().out


def test_hybrid_cli_exposes_stages_and_applies_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import swave.hybrid_runner as runner_module

    captured: list[HybridInversionConfig] = []
    tuning = tmp_path / "tuning.json"
    tuning.write_text("{}", encoding="utf-8")

    def tune(config: HybridInversionConfig) -> Path:
        captured.append(config)
        return tuning

    monkeypatch.setattr(runner_module, "select_prior_lambda", tune)
    config_path = tmp_path / "hybrid.toml"
    config_path.write_text("[hybrid]\nworkers = 1\n", encoding="utf-8")

    assert main(["hybrid-invert", "--help"]) == 0
    help_output = capsys.readouterr().out
    for option in (
        "--stage",
        "--dataset-config",
        "--dataset-dir",
        "--forward-checkpoint",
        "--supervised-dir",
        "--output-dir",
        "--device",
        "--workers",
        "--threads-per-worker",
        "--task-index",
        "--task-count",
    ):
        assert option in help_output

    code = main(
        [
            "hybrid-invert",
            "--config",
            str(config_path),
            "--stage",
            "tune",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--forward-checkpoint",
            str(tmp_path / "forward.pt"),
            "--supervised-dir",
            str(tmp_path / "supervised"),
            "--output-dir",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--workers",
            "3",
            "--threads-per-worker",
            "2",
            "--task-index",
            "1",
            "--task-count",
            "4",
        ]
    )

    assert code == 0
    assert capsys.readouterr().out.strip() == str(tuning)
    assert len(captured) == 1
    assert captured[0].dataset_dir == tmp_path / "dataset"
    assert captured[0].forward_checkpoint == tmp_path / "forward.pt"
    assert captured[0].supervised_dir == tmp_path / "supervised"
    assert captured[0].output_dir == tmp_path / "results"
    assert (captured[0].workers, captured[0].threads_per_worker) == (3, 2)
    assert (captured[0].task_index, captured[0].task_count) == (1, 4)


def test_hybrid_report_cli_passes_all_artifact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import swave.hybrid_report as report_module

    captured: list[tuple[Path, Path, Path, Path | None, Path | None, bool]] = []

    def build(
        results,
        dataset,
        output,
        *,
        baseline_summary,
        supervised_evaluation,
        allow_archived_software,
    ):
        captured.append(
            (
                results,
                dataset,
                output,
                baseline_summary,
                supervised_evaluation,
                allow_archived_software,
            )
        )
        return {"status": "ok"}

    monkeypatch.setattr(report_module, "build_hybrid_report", build)
    code = main(
        [
            "hybrid-report",
            "--results-dir",
            str(tmp_path / "results"),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "report"),
            "--baseline-summary",
            str(tmp_path / "baseline.json"),
            "--supervised-evaluation",
            str(tmp_path / "supervised.json"),
            "--allow-archived-software",
        ]
    )

    assert code == 0
    assert captured == [
        (
            tmp_path / "results",
            tmp_path / "dataset",
            tmp_path / "report",
            tmp_path / "baseline.json",
            tmp_path / "supervised.json",
            True,
        )
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_invert_applies_overrides_and_prints_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import swave.inversion_runner as runner_module

    captured: list[tuple[InversionConfig, str]] = []
    manifest = ResultManifest(
        schema_version=3,
        dataset_config_hash="a" * 64,
        dataset_manifest_sha256="d" * 64,
        checkpoint_sha256="b" * 64,
        split_policy=SPLIT_POLICY,
        inversion_config_hash="c" * 64,
        minimum_valid_solutions=None,
        experiment="full",
        expected_jobs=("full-clean-shard-00000",),
        expected_job_sample_count={"full-clean-shard-00000": 1},
        expected_job_sample_id_sha256={"full-clean-shard-00000": "e" * 64},
        completed_jobs=(),
        job_sha256={},
        package_version="0.1.0",
        software_sha256="f" * 64,
        created_at="2026-08-02T00:00:00+00:00",
        complete=False,
    )

    def run(config: InversionConfig, experiment: str) -> ResultManifest:
        captured.append((config, experiment))
        return manifest

    monkeypatch.setattr(runner_module, "run_inversion_experiment", run)
    config_path = tmp_path / "inversion.toml"
    config_path.write_text(
        "[inversion]\nworkers = 1\ntask_index = 0\ntask_count = 1\n",
        encoding="utf-8",
    )

    code = main(
        [
            "invert",
            "--config",
            str(config_path),
            "--experiment",
            "full",
            "--dataset-config",
            str(tmp_path / "dataset.toml"),
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--output-dir",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--workers",
            "3",
            "--deep-samples-per-job",
            "7",
            "--threads-per-worker",
            "2",
            "--task-index",
            "2",
            "--task-count",
            "4",
        ]
    )

    assert code == 0
    assert len(captured) == 1
    config, experiment = captured[0]
    assert experiment == "full"
    assert config == InversionConfig(
        dataset_config=tmp_path / "dataset.toml",
        dataset_dir=tmp_path / "dataset",
        checkpoint=tmp_path / "best.pt",
        output_dir=tmp_path / "results",
        device="cpu",
        workers=3,
        deep_samples_per_job=7,
        threads_per_worker=2,
        task_index=2,
        task_count=4,
    )
    expected_json = json.loads(json.dumps(asdict(manifest)))
    assert json.loads(capsys.readouterr().out) == expected_json


def test_invert_revalidates_combined_cluster_overrides(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "inversion.toml"
    config_path.write_text("[inversion]\n", encoding="utf-8")

    code = main(
        [
            "invert",
            "--config",
            str(config_path),
            "--task-index",
            "4",
            "--task-count",
            "4",
        ]
    )

    assert code == 2
    assert capsys.readouterr().err.startswith("error:")


def test_inversion_report_passes_directories_and_prints_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import swave.inversion_report as report_module

    captured: list[tuple[Path, Path, Path, DatasetConfig, bool]] = []

    def build(
        results: Path,
        dataset: Path,
        output: Path,
        *,
        dataset_config: DatasetConfig,
        allow_archived_software: bool,
    ) -> dict[str, object]:
        captured.append(
            (results, dataset, output, dataset_config, allow_archived_software)
        )
        return {"schema_version": 1, "value": 1.25, "missing": None}

    monkeypatch.setattr(report_module, "build_inversion_report", build)
    results = tmp_path / "results"
    dataset = tmp_path / "dataset"
    output = tmp_path / "report"
    dataset_config_path = tmp_path / "dataset.toml"
    dataset_config_path.write_text(
        "[physics]\nfmin = 1.0\nfmax = 120.0\nfstep = 1.0\n",
        encoding="utf-8",
    )

    code = main(
        [
            "inversion-report",
            "--dataset-config",
            str(dataset_config_path),
            "--results-dir",
            str(results),
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
            "--allow-archived-software",
        ]
    )

    assert code == 0
    assert captured == [
        (
            results,
            dataset,
            output,
            DatasetConfig(physics=PhysicsConfig(fmin=1.0, fmax=120.0, fstep=1.0)),
            True,
        )
    ]
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "value": 1.25,
        "missing": None,
    }


def test_readme_documents_exact_external_and_four_task_workflows() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

    for command in (
        "swave generate --config configs/dataset.toml",
        "swave train --config configs/training-48g.toml",
        "swave invert --config configs/inversion.toml --experiment both",
        "--results-dir results/inversion",
        "--dataset-dir data/production",
        "--output-dir results/inversion-report",
    ):
        assert command in readme
    for index in range(4):
        assert (
            "swave invert --config configs/inversion.toml --experiment both "
            f"--task-index {index} --task-count 4"
        ) in readme
    assert "mod100-v2-80-5-5-10" in readme
    assert "1% relative Gaussian noise" in readme


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


def test_train_help_exposes_data_loader_worker_override(capsys) -> None:
    assert main(["train", "--help"]) == 0
    assert "--num-workers" in capsys.readouterr().out


def test_invalid_predict_input_returns_clean_error(tmp_path: Path, capsys) -> None:
    code = main(["predict", str(tmp_path / "missing.pt"), "0.5"])
    assert code == 2
    assert capsys.readouterr().err.startswith("error:")
