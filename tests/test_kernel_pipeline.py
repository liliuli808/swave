from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_prefers_checkout_over_a_stale_pythonpath_package(tmp_path: Path):
    repository = Path(__file__).resolve().parents[1]
    stale = tmp_path / "old-installation" / "swave"
    stale.mkdir(parents=True)
    (stale / "__init__.py").write_text(
        'raise RuntimeError("stale swave package was imported")\n'
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for index in range(100):
        (dataset / f"shard-{index:05d}.h5").touch()
    base = tmp_path / "base.pt"
    base.touch()
    corrections = tmp_path / "corrections.npz"
    corrections.touch()

    # Run the real source/import preflight, then stop at the test stage so no
    # dataset generation or nested pytest invocation takes place.
    interpreter = tmp_path / "preflight-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pytest']:\n"
        "    sys.exit(77)\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", str(repository / "scripts/run_kernel_refinement.sh")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHON": str(interpreter),
            "PYTHONPATH": str(stale.parent),
            "DATASET_DIR": str(dataset),
            "BASE_CHECKPOINT": str(base),
            "CORRECTIONS": str(corrections),
            "OUTPUT_DIR": str(tmp_path / "run"),
            "DEVICE": "cpu",
            "REQUIRE_WARM_START": "0",
        },
        text=True, capture_output=True, timeout=60, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "kernel/solver tests failed" in result.stdout
    assert "stale swave package was imported" not in result.stderr
    record = next(json.loads(line) for line in result.stdout.splitlines()
                  if line.startswith("{"))
    assert record["kernel_training_source"] == str(
        repository / "src/swave/kernel_training.py"
    )
    assert record["layer_norm"] == "HigherOrderLayerNorm"


def test_row_experiment_passes_the_controlled_configuration_and_pins_source(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    interpreter = tmp_path / "record-python"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1:3] == ['-m', 'pytest']:\n"
        "    sys.exit(0)\n"
        "print(json.dumps({'args': sys.argv[1:], 'path': os.environ['PYTHONPATH']}))\n"
    )
    interpreter.chmod(0o755)
    result = subprocess.run(
        ["bash", str(repository / "scripts/run_kernel_row_refinement.sh")],
        cwd=tmp_path, text=True, capture_output=True, timeout=30, check=False,
        env={
            **os.environ, "PYTHON": str(interpreter), "DEVICE": "cpu",
            "BASE_CHECKPOINT": "known-base.pt", "OUTPUT_DIR": "new-run",
            "KERNEL_ROW_WEIGHT": "0.2", "KERNEL_ROW_BATCH_SIZE": "64",
            "KERNEL_MINING_SAMPLES": "32", "KERNEL_MINING_INTERVAL": "3",
            "EPOCHS": "12", "PYTHONPATH": "/tmp/stale-swave",
        },
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["path"].split(":")[0] == str(repository / "src")
    args = record["args"]
    for name, value in {
        "--base-checkpoint": "known-base.pt", "--output-dir": "new-run",
        "--kernel-row-weight": "0.2", "--kernel-row-batch-size": "64",
        "--kernel-mining-samples": "32", "--kernel-mining-interval": "3",
        "--kernel-weight": "1", "--kernel-directions": "2", "--epochs": "12",
    }.items():
        assert args[args.index(name) + 1] == value
    assert "--require-warm-start" in args
