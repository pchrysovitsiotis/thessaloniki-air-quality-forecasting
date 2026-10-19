# Reproduction and verification

## Offline demonstration

Run `python demo.py` with Python 3.10+. It reads only the included `examples/` and `results/` files and writes a standalone HTML page. No third-party package, trained model, dataset download, environment variable or API key is needed.

```sh
python demo.py --check
python -m unittest discover -s tests -v
```

The tests check numeric consistency, interval ordering, exact fact references, safe HTML rendering and execution of a copied demo with Python site packages disabled. The demo's report checks cover these fixed fixtures; the full reporting component uses `jsonschema` for JSON Schema validation. Neither form of checking certifies the meaning of arbitrary prose.

## Full reference pipeline

This path trains models and needs the archive described in [DATA.md](DATA.md). Use a separate Python 3.12 virtual environment. `requirements.txt` records the versions of the full research environment, including a CPU build of PyTorch; the demo does not need these packages.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` in a POSIX shell. Then:

```sh
python -m pip install -r requirements.txt
python run_pipeline.py --data-dir data --check-inputs
python run_pipeline.py --data-dir data --output-dir outputs/preparation --with-raw --prepare-only
```

Each invocation needs a new or empty output folder. Preparation builds large feature arrays and per-horizon copies; allow several GB of free disk space. `--prepare-only` completes input preparation and checks, with no fitting or API call.

To run the full forecasting experiment and raw-input comparison:

```sh
python run_pipeline.py --data-dir data --output-dir outputs/full_run --with-raw
```

This prepares both arms, trains the +1/+6/+12-hour models, evaluates them, verifies paired prediction targets, aggregates results and prepares the deterministic reporting evidence. It can take substantially longer than the demo. It **does not call Gemini by default**. Omit `--with-raw` to run only the processed arm.

The entry point copies required inputs into the chosen output directory. Generated models, logs, caches and reports stay there. `PM10_OUTPUT_DIR` provides this root to component scripts. The source tree and bundled examples are separate from generated artifacts.

## Generating new LLM responses

The offline demo always uses cached responses. New responses require a Google Gemini API key and transmit the prepared station/time/measurement/forecast evidence to Google. Model access and any applicable charges depend on the API account. `.env.example` contains an empty template; the code reads environment variables and does **not** automatically load `.env` files.

After a full forecasting run, reuse its outputs without training again. In PowerShell:

```powershell
$env:PM10_OUTPUT_DIR = (Resolve-Path 'outputs/full_run').Path
$env:GEMINI_API_KEY = 'YOUR_OWN_KEY'
python src/reports/code/app_reports.py
```

Use `--experiment` on the last command for the two-condition, three-repetition reporting experiment. Completed request responses are cached. Use `--prepare` to rebuild evidence and schemas without requesting new responses. Never commit a real key or shell transcript containing one.

Alternatively, `--llm-mode reports` or `--llm-mode experiment` on `run_pipeline.py` includes this stage in a **new full run**. The default is `--llm-mode prepare`.

Model-generated reports require review; new outputs are not automatically the edited examples bundled with this repository. The reference component's automatic checks concern structure and explicit evidence checks, not a complete assessment of meaning.

## Experimental interpretation

The fixed history is 24 hours. Direct horizons are +1, +6 and +12 hours. The seed is 42. ANN, TCN and XGBoost use the same processed information, followed by four non-negative, sum-to-one weighted ensembles. Persistence repeats the last retained PM10 measurement.

| Stage boundary | UTC |
|---|---|
| Inner validation starts | 23 June 2024, 16:00 |
| Meta starts | 5 November 2024, 15:00 |
| Calibration starts | 23 April 2025, 06:00 |
| Test starts | 13 August 2025, 17:00 |

Preprocessing statistics are fitted before inner validation. Training settings are selected on inner validation, then the base models are refitted on the full training period. Meta fits ensemble weights; calibration fits interval widths; test evaluates the final models. Boundary-crossing targets are purged. The historical test period had been inspected during development, which limits claims of independence.

The raw arm still requires hourly alignment, rejection of impossible numeric inputs, median imputation and scaling. It uses eleven original variables × 24 hours, with no engineered masks, flags or static context. Validation, meta, calibration and test cases match the processed arm. Training samples can differ. The before/after comparison therefore measures a combined intervention, not an isolated feature-engineering ablation.

The exported source is the current thesis pipeline with portable paths and a new entry point. The saved results were not retrained for this portfolio export. Re-running with different hardware or numerical-library versions may change results; exact bitwise reproduction is not promised. Model binaries and the full archive are not bundled, so a fresh clone supports the offline demo immediately, but not new model inference.

## Recreate the README figure

With Matplotlib available (it is included in the full requirements):

```sh
python tools/plot_results.py
```

The figure is generated from the included CSV metrics. It does not train models or alter recorded results.
