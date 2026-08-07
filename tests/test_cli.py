import numpy as np
import pytest

from chirp_gen.cli import main


def test_cli_writes_npz_with_default_gw150914_params(tmp_path):
    """CLI with defaults writes a loadable NPZ under the given output path."""
    out = tmp_path / "gw150914.npz"
    main(["--output", str(out)])

    assert out.exists()
    with np.load(out, allow_pickle=False) as data:
        assert len(data["time_s"]) > 1
        assert "strain" in data.files
        assert "metadata_json" in data.files


def test_cli_writes_csv_when_format_requested(tmp_path):
    """CLI --format csv writes a CSV even if the suffix is unusual."""
    out = tmp_path / "chirp.out"
    main(["--output", str(out), "--format", "csv"])

    text = out.read_text(encoding="utf-8")
    assert "# m1_msun=" in text
    data = np.loadtxt(out, delimiter=",", comments="#")
    assert data.shape[1] == 4


def test_cli_missing_output_exits_nonzero():
    """CLI without --output exits with a non-zero status via argparse."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code != 0


def test_cli_invalid_mass_ordering_exits_nonzero(tmp_path):
    """CLI exits non-zero when validation rejects m2 > m1."""
    out = tmp_path / "bad.npz"
    with pytest.raises(SystemExit) as exc_info:
        main(["--m1", "10", "--m2", "20", "--output", str(out)])
    assert exc_info.value.code == 1
    assert not out.exists()
