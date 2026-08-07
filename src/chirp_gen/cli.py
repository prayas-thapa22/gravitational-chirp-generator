"""Command-line interface for generating and saving chirp waveforms.

Defaults match approximate GW150914 values so a bare ``chirp-gen -o out.npz``
run is a meaningful sanity check (inspiral band ~35-250 Hz at 410 Mpc).
"""

from __future__ import annotations

import argparse
import sys

from chirp_gen.io import save_waveform
from chirp_gen.models import BinaryParameters, WaveformConfig
from chirp_gen.pipeline import generate_waveform


def _build_parser() -> argparse.ArgumentParser:
    """Return the argparse parser for the chirp-gen CLI."""
    parser = argparse.ArgumentParser(
        prog="chirp-gen",
        description=(
            "Generate a Newtonian quadrupole inspiral chirp and save it to "
            "NPZ or CSV. Defaults are GW150914-like."
        ),
    )
    parser.add_argument(
        "--m1",
        type=float,
        default=36.0,
        help="Primary mass in solar masses (default: 36)",
    )
    parser.add_argument(
        "--m2",
        type=float,
        default=29.0,
        help="Secondary mass in solar masses (default: 29)",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=410.0,
        help="Luminosity distance in megaparsecs (default: 410)",
    )
    parser.add_argument(
        "--f-start",
        type=float,
        default=35.0,
        dest="f_start",
        help="Band start frequency in Hertz (default: 35)",
    )
    parser.add_argument(
        "--f-end",
        type=float,
        default=250.0,
        dest="f_end",
        help="Band end frequency in Hertz (default: 250)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=4096.0,
        dest="sample_rate",
        help="Sample rate in Hertz (default: 4096)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path (.npz or .csv)",
    )
    parser.add_argument(
        "--format",
        choices=("npz", "csv"),
        default=None,
        help="File format (default: infer from output suffix)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the chirp-gen CLI: generate a waveform and save it to disk.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: On argparse errors, validation failures, or I/O issues.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    params = BinaryParameters(
        m1_msun=args.m1,
        m2_msun=args.m2,
        distance_mpc=args.distance,
        f_start_hz=args.f_start,
        f_end_hz=args.f_end,
    )
    config = WaveformConfig(sample_rate_hz=args.sample_rate)

    try:
        result = generate_waveform(params, config)
        out = save_waveform(result, args.output, format=args.format)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Wrote {len(result.time_s)} samples to {out}")


if __name__ == "__main__":
    main()
