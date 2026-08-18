# CAIS-Stability

Reference implementation and reproducible benchmark suite for **governance stability, recoverability, and risk-bounded autonomy in Controlled Agentic AI Systems (CAIS)**.

The repository evaluates trajectory-level governance across maritime, supply-chain, and smart-grid benchmark environments, with a shared governance core and common metrics.

## Status

Initial research scaffold. The public API and benchmark configuration may evolve before the first archival release.

## Research questions

1. Can dynamic CAIS governance improve recoverability while preserving hard safety compared with ungoverned, static-governance, and binary runtime-assurance baselines?
2. Under what conditions can governance stability and finite-time recovery be maintained under bounded runtime disturbances?
3. What is the trade-off between safety, recoverability, task utility, and retained autonomy?
4. Are governance stability and recoverability transferable across heterogeneous domains without changing the core governance mechanism?

## Planned modes

- `AUTO` — execute the agent proposal unchanged.
- `CONSTRAIN` — project or modify a risky proposal into the admissible action set.
- `RECOVER` — prioritize finite-time return toward the nominal set.
- `FALLBACK` — execute a conservative safe fallback action.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy src
pytest -q
python scripts/verify.py
```

## License

MIT.
