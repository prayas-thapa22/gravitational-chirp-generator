"""Save gravitational-wave waveforms to disk for reuse and plotting.

Persists ``WaveformResult`` arrays together with run metadata so a saved
file remains reproducible (masses, distance, band, sample rate).

Supported formats:

- NPZ: compressed NumPy archive (full float precision).
- CSV: human-readable table with ``# key=value`` metadata header.

Strain is dimensionless; metadata is required to interpret the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chirp_gen.models import WaveformResult


def _resolve_format(path: Path, format: str | None) -> str:
    """Return normalized format string ('npz' or 'csv').

    Args:
        path: Output path (used to infer format from suffix when needed).
        format: Explicit format, or None to infer from ``path.suffix``.

    Returns:
        Either ``"npz"`` or ``"csv"``.

    Raises:
        ValueError: If the format cannot be resolved or is unsupported.
    """
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return "npz"
        if suffix == ".csv":
            return "csv"
        raise ValueError(
            f"Cannot infer format from path '{path}'; "
            "use a .npz/.csv suffix or pass format='npz'|'csv'"
        )

    normalized = format.lower().strip()
    if normalized in ("npz", "csv"):
        return normalized
    raise ValueError(f"Unsupported format '{format}'; use 'npz' or 'csv'")


def _save_npz(result: WaveformResult, path: Path) -> None:
    """Write arrays and metadata JSON into a compressed NPZ archive."""
    np.savez_compressed(
        path,
        time_s=result.time_s,
        strain=result.strain,
        frequency_hz=result.frequency_hz,
        phase_rad=result.phase_rad,
        metadata_json=np.asarray(json.dumps(result.metadata)),
    )


def _save_csv(result: WaveformResult, path: Path) -> None:
    """Write metadata header comments plus a four-column CSV table."""
    lines = [f"# {key}={value}" for key, value in result.metadata.items()]
    lines.append("# columns: time_s,strain,frequency_hz,phase_rad")
    header = "\n".join(lines)

    data = np.column_stack(
        (
            result.time_s,
            result.strain,
            result.frequency_hz,
            result.phase_rad,
        )
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.18e",
    )


def save_waveform(
    result: WaveformResult,
    path: str | Path,
    format: str | None = None,
) -> Path:
    """Save a waveform result to NPZ or CSV, including run metadata.

    Args:
        result: Generated waveform arrays and metadata.
        path: Output file path (``.npz`` or ``.csv``).
        format: ``"npz"`` or ``"csv"``. If None, inferred from ``path`` suffix.

    Returns:
        Resolved ``Path`` written to disk.

    Raises:
        ValueError: If the format is missing or unsupported.
    """
    out = Path(path)
    resolved = _resolve_format(out, format)
    out.parent.mkdir(parents=True, exist_ok=True)

    if resolved == "npz":
        _save_npz(result, out)
    else:
        _save_csv(result, out)

    return out.resolve()
