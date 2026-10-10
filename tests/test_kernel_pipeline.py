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
