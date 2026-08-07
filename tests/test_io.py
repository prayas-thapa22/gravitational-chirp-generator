import json

import numpy as np
import pytest

from chirp_gen.io import save_waveform
from chirp_gen.models import BinaryParameters, WaveformConfig
from chirp_gen.pipeline import generate_waveform


@pytest.fixture
def gw150914_result():
    """GW150914-like WaveformResult for I/O round-trip tests."""
    params = BinaryParameters(
        m1_msun=36.0,
        m2_msun=29.0,
        distance_mpc=410.0,
        f_start_hz=35.0,
        f_end_hz=250.0,
    )
    config = WaveformConfig(sample_rate_hz=4096.0)
    return generate_waveform(params, config)


def test_save_waveform_npz_round_trip(gw150914_result, tmp_path):
    """NPZ save preserves arrays and recovers metadata keys."""
    path = tmp_path / "chirp.npz"
    saved = save_waveform(gw150914_result, path)

    assert saved.exists()
    with np.load(saved, allow_pickle=False) as data:
        assert np.allclose(data["time_s"], gw150914_result.time_s)
        assert np.allclose(data["strain"], gw150914_result.strain)
        assert np.allclose(data["frequency_hz"], gw150914_result.frequency_hz)
        assert np.allclose(data["phase_rad"], gw150914_result.phase_rad)
        meta = json.loads(data["metadata_json"].item())
    assert meta["m1_msun"] == gw150914_result.metadata["m1_msun"]
    assert meta["f_end_hz"] == gw150914_result.metadata["f_end_hz"]
    assert "M_c_kg" in meta


def test_save_waveform_csv_shape_and_metadata_header(gw150914_result, tmp_path):
    """CSV has four columns, matching row count, and metadata in # headers."""
    path = tmp_path / "chirp.csv"
    saved = save_waveform(gw150914_result, path, format="csv")

    text = saved.read_text(encoding="utf-8")
    assert "# m1_msun=" in text
    assert "# sample_rate_hz=" in text
    assert "# columns: time_s,strain,frequency_hz,phase_rad" in text

    data = np.loadtxt(saved, delimiter=",", comments="#")
    assert data.ndim == 2
    assert data.shape[1] == 4
    assert data.shape[0] == len(gw150914_result.time_s)
    assert np.allclose(data[:, 0], gw150914_result.time_s)
    assert np.allclose(data[:, 1], gw150914_result.strain)


def test_save_waveform_rejects_unsupported_format(gw150914_result, tmp_path):
    """Unsupported format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported format"):
        save_waveform(gw150914_result, tmp_path / "out.dat", format="wav")
