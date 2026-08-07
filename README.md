# gravitational-chirp-generator

A Python library for generating synthetic gravitational-wave chirps from binary black hole inspirals using the quadrupole formula in the Newtonian limit. The package provides a typed library API and a `chirp-gen` CLI for producing time-domain strain waveforms.

## Setup

The package lives under `src/`, so install it (editable) before importing `chirp_gen` or running tests:

```bash
pip install -e ".[dev]"
```

This installs runtime dependencies (`numpy`, `scipy`) plus dev tools (`pytest`, `ruff`, `matplotlib`).

### Run tests

```bash
pytest -v
```

Without the editable install, pytest fails with `ModuleNotFoundError: No module named 'chirp_gen'`.

## Quickstart (library)

```python
from chirp_gen import BinaryParameters, WaveformConfig, generate_waveform, save_waveform

params = BinaryParameters(
    m1_msun=36.0,
    m2_msun=29.0,
    distance_mpc=410.0,
    f_start_hz=35.0,
    f_end_hz=250.0,
)
config = WaveformConfig(sample_rate_hz=4096.0)

result = generate_waveform(params, config)
save_waveform(result, "out/gw150914.npz")
```

## CLI

Defaults are GW150914-like (`36`/`29` Msun, `410` Mpc, `35`–`250` Hz, `4096` Hz sample rate):

```bash
chirp-gen -o out/gw150914.npz
```

Optional flags: `--m1`, `--m2`, `--distance`, `--f-start`, `--f-end`, `--sample-rate`, `--format npz|csv`.

## Visualization

Open [`notebooks/01_chirp_visualization.ipynb`](notebooks/01_chirp_visualization.ipynb) in VS Code / Cursor, or install Jupyter Lab separately and launch it:

```bash
pip install jupyterlab
jupyter lab notebooks/01_chirp_visualization.ipynb
```

You should see:

- strain $h(t)$ getting faster and louder toward merger ($t \to 0$)
- frequency $f(t)$ rising through the band
- a heavier chirp mass sweeping the same band in less time

## Physics assumptions (MVP)

**Included:** circular binary inspiral, leading-order quadrupole radiation, single-polarization face-on strain, band-limited segment with merger at $t = 0$.

**Permanent limits (not deferred features):** no merger or ringdown, no spin or eccentricity, no post-Newtonian corrections, no detector response, and no noise. Detection / parameter estimation belong in a separate analysis project that can depend on this package.

## Project structure

```text
gravitational-chirp-generator/
├── .github/workflows/   # CI (pytest + ruff)
├── src/chirp_gen/       # Installable package
│   ├── constants.py     # Physical constants and unit conversions
│   ├── models.py        # Input/output dataclasses
│   ├── physics.py       # Core quadrupole inspiral equations
│   ├── pipeline.py      # generate_waveform orchestration
│   ├── io.py            # Save waveforms (CSV, NPZ)
│   └── cli.py           # chirp-gen command-line interface
├── notebooks/           # Chirp visualization
├── tests/               # Unit and integration tests
├── pyproject.toml       # Packaging and dependencies
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
