# gravitational-chirp-generator

A Python library for generating synthetic gravitational-wave chirps from binary black hole inspirals using the quadrupole formula in the Newtonian limit. The goal is a small, production-ready package with both a library API and a command-line tool for producing time-domain strain waveforms.

## Project structure

```text
gravitational-chirp-generator/
├── src/chirp_gen/       # Installable package
│   ├── constants.py     # Physical constants and unit conversions
│   ├── models.py        # Input/output dataclasses (masses, config, waveform result)
│   ├── physics.py       # Core quadrupole inspiral equations
│   ├── pipeline.py      # High-level waveform generation orchestration
│   ├── io.py            # Save/load waveforms (CSV, NPZ)
│   └── cli.py           # Command-line interface
├── tests/               # Unit and integration tests
├── pyproject.toml       # Packaging and dependencies
└── requirements.txt
```

## Implementation plan

1. **Constants & models** — Define SI constants (`G`, `c`, `M_sun`) and typed parameter objects for binary masses, distance, and waveform configuration.
2. **Physics** — Implement quadrupole inspiral: chirp mass, frequency evolution `f(t)`, phase, amplitude, and time-domain strain `h(t)`.
3. **Pipeline** — Add `generate_waveform()` with input validation and a uniform time grid.
4. **I/O & CLI** — Export waveforms to file and expose a `chirp-gen` command-line entry point.
5. **Tests & docs** — Add analytic unit tests, example scripts, and document physics assumptions and limitations.

### Scope (MVP)

- **Included:** circular binary inspiral, quadrupole radiation, single-polarization strain
- **Not included (yet):** merger, ringdown, spin, eccentricity, post-Newtonian corrections, detector response, noise
