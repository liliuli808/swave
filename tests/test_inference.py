from pathlib import Path

import numpy as np
import pytest
import torch

from swave.inference import ForwardPredictor
from swave.network import FourHeadForwardModel


def _checkpoint(path: Path) -> Path:
    model = FourHeadForwardModel()
    torch.save(
        {
            "model": model.state_dict(),
            "input_mean": np.ones(20, dtype=np.float32),
            "input_std": np.ones(20, dtype=np.float32),
            "target_mean": np.zeros((4, 1), dtype=np.float32),
            "target_std": np.ones((4, 1), dtype=np.float32),
        },
        path,
    )
    return path


def test_predict_supports_single_and_batched_vs(tmp_path: Path) -> None:
    predictor = ForwardPredictor.load(
        _checkpoint(tmp_path / "model.pt"), device="cpu"
    )
    single = np.linspace(0.3, 2.0, 20)
    assert predictor.predict(single).shape == (4, 120)
    assert predictor.predict(np.stack([single, single])).shape == (2, 4, 120)
    frequencies, curves = predictor.predict_with_frequencies(single)
    np.testing.assert_allclose(frequencies[[0, -1]], [0.5, 60.0])
    assert curves.shape == (4, 120)


def test_predict_rejects_nonfinite_or_out_of_range_vs(tmp_path: Path) -> None:
    predictor = ForwardPredictor.load(
        _checkpoint(tmp_path / "model.pt"), device="cpu"
    )
    invalid = np.linspace(0.3, 2.0, 20)
    invalid[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predictor.predict(invalid)
    with pytest.raises(ValueError, match="0.3"):
        predictor.predict(np.full(20, 2.7))
