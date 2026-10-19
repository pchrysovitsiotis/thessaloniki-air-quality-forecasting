# Data and examples

## Included in this repository

`examples/evidence.json` is one historical snapshot containing observations, stored forecasts, estimated intervals, time labels and deterministic facts used by the reporting component. It contains no future observed targets. `examples/provenance.json` records its selection and limitations.

The original LLM response texts, their parsed JSON outputs, schemas and a small comparison summary are included. Review notes distinguish edited examples from original completions. The original-completion hash in each review note identifies the corresponding JSON file in `examples/original/`.

The result CSV files contain complete-test summaries, station summaries and the paired raw-versus-processed comparison. They are not the underlying per-observation predictions. The demo displays their saved values; it does not recalculate forecasting metrics.

`examples/manifest.json` records SHA-256 hashes for all files under `examples/` and `results/`, except itself. This detects accidental fixture changes, not independent scientific validity or authorship. `.gitattributes` preserves fixture bytes so checksums also work after a Windows/Linux clone.

## Local archive required for the full pipeline

The complete pollutant/weather archive is not redistributed here, and no automatic download URL is supplied. The reference pipeline expects the exact local archive used in the thesis. An arbitrary EEA download is not a drop-in replacement.

Prepare this layout in a separate directory, or in the ignored repository `data/` folder:

```text
data/
  source/
    original_pollutants.parquet
    station_weather.parquet
    station_metadata.json
  metadata/
    PM10_metadata.csv
```

| File | Expected content |
|---|---|
| `original_pollutants.parquet` | Long-format `station`, `datetime`, `pollutant`, `value` columns. Original hourly observations before the project's quality checks; no EEA validity fields are read. |
| `station_weather.parquet` | `station_id`, UTC `datetime`, `temperature_2m`, `relative_humidity_2m`, `rain`, `surface_pressure`, `wind_speed_10m`, `wind_speed_100m`, `wind_direction_10m`. Wind direction is present in the archive but is not one of the eleven model input variables. |
| `station_metadata.json` | Mapping from station ID to numeric `latitude` and `longitude`. |
| `PM10_metadata.csv` | Reporting metadata. Required columns: `Sampling Point Id`, `Air Quality Station Type`, `Operational Activity Begin`. Dates are parsed day-first. Every report station must have a matching sampling-point row. |

The archived pollutant variables used by the models are PM10, PM2.5, NO₂, O₃ and CO. The reporting snapshot contains only the available, retained values needed by its report tasks. Reporting labels use station type from the metadata CSV; the coordinates used in modelling come from the JSON file.

Three source-file hashes are stored in `configs/forecasting/source_manifest.json`. The entry point verifies them before any preparation. Reporting metadata is checked for required columns and station coverage, but is not included in those three recorded hashes. A different archive requires an explicit data-adaptation and validation exercise, including a new manifest and appropriate split dates; changing hashes alone does not establish comparability.

Use data only under the applicable source terms. This repository does not grant rights over a third-party archive. Detailed scientific assumptions and fixed calendar boundaries are available in `configs/forecasting/protocol.json` and `config.json`.

## Quality checks versus reporting guidance

Data cleaning is independent of EEA validity flags. It uses transparent local rules, causal checks and thresholds fitted on the training portion. Statistical anomaly flags are not certified instrument validation.

The reporting component separately applies a recorded snapshot of the [EEA air-quality index methodology](https://airindex.eea.europa.eu/AQI/index.html) to available measured pollutants and uses Greek paraphrases of its general guidance. The methodology snapshot was checked on 9 October 2026; it is not evidence of a live feed. PM10 forecasts do not determine a future overall index. Missing pollutant coverage remains explicit.
