"""Offline replay of saved PM10 predictions and reports. Python standard library only."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
KINDS = {"evolution": "PM10 outlook", "activities": "Outdoor activities", "comparison": "Station comparison"}
ENGLISH_NAMES = {"GR0018A": "Agia Sofia", "GR0019A": "Kalamaria", "GR0020A": "Kordelio", "GR0046A": "Sindos"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    result = float(str(value).replace(",", "."))
    require(math.isfinite(result), "Non-finite example value")
    return result


def verify_files(root):
    """Detect missing or changed fixtures; hashes are not a provenance signature."""
    manifest = read_json(root / "examples/manifest.json")
    actual = {p.relative_to(root).as_posix() for folder in ("examples", "results")
              for p in (root / folder).rglob("*") if p.is_file() and p.name != "manifest.json"}
    require(actual == set(manifest["files"]), "Example file inventory differs from the manifest")
    for relative, expected in manifest["files"].items():
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "Invalid fixture path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "Fixture checksum mismatch: " + relative)


def validate_evidence(evidence):
    """Check the fixed historical example, including forecast times and displayed deltas."""
    require(evidence["mode"] == "historical_replay", "Expected a historical replay")
    require(evidence["history_hours"] == 24, "Expected 24-hour history")
    require(evidence["forecast_horizons"] == [1, 6, 12], "Unexpected forecast horizons")
    stations = evidence["stations"]
    require(len(stations) == 4 and {s["id"] for s in stations} == set(ENGLISH_NAMES), "Unexpected station coverage")
    origin = datetime.strptime(evidence["origin_local"], "%d/%m/%Y %H:%M")
    for station in stations:
        observed = number(station["observed_pm10"])
        require(observed >= 0, "Negative observation")
        require(station["time"] == evidence["origin_local"], "Stations have different origins")
        forecasts = station["forecasts"]
        require([f["horizon"] for f in forecasts] == [1, 6, 12], "Missing or reordered forecast")
        require(abs(number(evidence["facts"][station["id"] + ".observed"]) - observed) <= .051, "Observed fact mismatch")
        for f in forecasts:
            value, lower, upper = (number(f[key]) for key in ("pm10", "lower", "upper"))
            require(0 <= lower <= value <= upper, "Invalid prediction interval")
            expected_time = (origin + timedelta(hours=f["horizon"])).strftime("%d/%m/%Y %H:%M")
            require(f["time"] == expected_time, "Forecast time does not match its horizon")
            require(abs(value - observed - number(f["difference"])) <= .11, "Forecast change does not match observation")
            require(evidence["facts"][station["id"] + f'.h{f["horizon"]}'] == f["pm10"], "Forecast fact mismatch")


def validate_report(report, schema, facts):
    """Checks for these fixed report fixtures; not a general JSON Schema implementation."""
    require(set(report) == {"title", "sections", "evidence_checks"}, "Unexpected report fields")
    require(isinstance(report["title"], str) and report["title"].strip(), "Empty report title")
    headings = schema["properties"]["sections"]["items"]["properties"]["heading"]["enum"]
    require(len(report["sections"]) == 3, "Expected three report sections")
    require([s["heading"] for s in report["sections"]] == headings, "Report headings are out of order")
    for section in report["sections"]:
        require(set(section) == {"heading", "text"}, "Unexpected section fields")
        require(isinstance(section["text"], str) and section["text"].strip(), "Empty section")
    checks = report["evidence_checks"]
    require(len(checks) >= 4 and len({c["fact_id"] for c in checks}) >= 4, "Insufficient distinct fact references")
    for item in checks:
        require(set(item) == {"fact_id", "value"}, "Unexpected fact-reference fields")
        require(item["fact_id"] in facts and facts[item["fact_id"]] == item["value"], "Invalid fact reference")


def load_bundle(root=ROOT):
    verify_files(root)
    evidence = read_json(root / "examples/evidence.json")
    validate_evidence(evidence)
    reports, originals = {}, {}
    for kind in KINDS:
        schema = read_json(root / f"examples/contracts/{kind}.json")
        reports[kind] = read_json(root / f"examples/reviewed/{kind}.json")
        validate_report(reports[kind], schema, evidence["facts"])
        originals[kind] = []
        for repetition in (1, 2, 3):
            plain = (root / f"examples/original/{kind}_plain_r{repetition}.txt").read_text(encoding="utf-8")
            contract = (root / f"examples/original/{kind}_contract_r{repetition}.txt").read_text(encoding="utf-8")
            parsed = read_json(root / f"examples/original/{kind}_contract_r{repetition}.json")
            require(json.loads(contract) == parsed, "Raw response and parsed JSON disagree")
            validate_report(parsed, schema, evidence["facts"])
            originals[kind].append({"plain": plain, "contract": contract})
    comparison = read_json(root / "examples/contract_comparison.json")
    require(len(comparison) == 6 and {(r["kind"], r["mode"]) for r in comparison}
            == {(k, m) for k in KINDS for m in ("plain", "contract")}, "Unexpected contract comparison")
    require(all(r["n"] == 3 for r in comparison), "Expected three repetitions per condition")
    with (root / "results/processed_test_metrics.csv").open(encoding="utf-8", newline="") as f:
        metrics = list(csv.DictReader(f))
    require(len(metrics) == 24 and all(r["scope"] == "all_test" for r in metrics), "Expected complete test metrics")
    return {"evidence": evidence, "reports": reports, "originals": originals,
            "comparison": comparison, "metrics": metrics}


def station_chart(station):
    """Discrete points and interval whiskers, not a continuous forecast curve."""
    forecasts = station["forecasts"]
    values = [number(f["pm10"]) for f in forecasts]
    upper = max(100, math.ceil(max(number(f["upper"]) for f in forecasts) / 20) * 20)
    y = lambda value: 128 - 100 * value / upper
    bits = [f'<svg viewBox="0 0 340 160" role="img" aria-label="{escape(ENGLISH_NAMES[station["id"]])}: discrete PM10 forecasts with estimated intervals">']
    for value in (0, upper / 2, upper):
        bits.append(f'<line x1="32" x2="323" y1="{y(value)}" y2="{y(value)}" stroke="#e4e9eb"/><text x="3" y="{y(value)+4}" fill="#61747c" font-size="11">{value:g}</text>')
    for x, f, value in zip((85, 180, 275), forecasts, values):
        low, high = y(number(f["lower"])), y(number(f["upper"]))
        bits.append(f'<path d="M{x},{low}V{high}M{x-7},{low}H{x+7}M{x-7},{high}H{x+7}" stroke="#679992" stroke-width="2" fill="none"/>')
        bits.append(f'<circle cx="{x}" cy="{y(value)}" r="5" fill="#087e75"/><text x="{x}" y="151" text-anchor="middle" fill="#364c58" font-size="12">+{f["horizon"]} h</text>')
    return "".join(bits) + "</svg>"


def report_html(report):
    body = ''.join(f'<h4>{escape(s["heading"])}</h4><p>{escape(s["text"])}</p>' for s in report["sections"])
    return f'<div lang="el" class="prose"><h3>{escape(report["title"])}</h3>{body}</div>'


def render_html(bundle):
    evidence = bundle["evidence"]
    cards = []
    for s in evidence["stations"]:
        cells = ''.join(f'<tr><td>+{f["horizon"]} h</td><td>{number(f["pm10"]):.1f}</td><td>{number(f["lower"]):.1f}–{number(f["upper"]):.1f}</td><td>{escape(f["time"])}</td></tr>' for f in s["forecasts"])
        cards.append(f'<article class="card"><div class="eyebrow">{escape(s["station_type"])} station</div><h3>{escape(ENGLISH_NAMES[s["id"]])} <span lang="el">{escape(s["name"])}</span></h3><div class="observation">{number(s["observed_pm10"]):.1f} <small>μg/m³ observed at origin</small></div>{station_chart(s)}<div class="scroll"><table><thead><tr><th>Horizon</th><th>Forecast</th><th>Interval</th><th>Local time</th></tr></thead><tbody>{cells}</tbody></table></div></article>')
    report_panels = []
    for kind, title in KINDS.items():
        saved = []
        for i, pair in enumerate(bundle["originals"][kind], 1):
            saved.append(f'<details><summary>Repetition {i} · original responses</summary><div class="comparison-grid"><div><h4>Free text</h4><pre lang="el">{escape(pair["plain"])}</pre></div><div><h4>JSON contract</h4><pre lang="el">{escape(pair["contract"])}</pre></div></div></details>')
        report_panels.append(f'<article class="report"><div class="eyebrow">Saved report</div><h3>{title}</h3><details><summary>Edited example · AI-assisted review</summary>{report_html(bundle["reports"][kind])}</details>{"".join(saved)}</article>')
    comparison_rows = []
    for row in bundle["comparison"]:
        schema = f'{row["schema_pass"]}/3' if row["mode"] == "contract" else "Not requested"
        comparison_rows.append(f'<tr><td>{KINDS[row["kind"]]}</td><td>{"JSON contract" if row["mode"] == "contract" else "Free text"}</td><td>{schema}</td><td>{row["length_pass"]}/3</td><td>{row["mean_words"]:g}</td></tr>')
    metric_rows = ''.join(f'<tr><td>+{r["horizon"]} h</td><td>{escape(r["model"].replace("_", " + "))}</td><td>{float(r["RMSE"]):.2f}</td><td>{float(r["MAE"]):.2f}</td><td>{float(r["R2"]):.3f}</td><td>{int(r["n"]):,}</td></tr>' for r in bundle["metrics"])
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>Thessaloniki · Air-quality forecasting</title>
<style>
:root{color-scheme:light;--ink:#183342;--muted:#536874;--teal:#087e75;--line:#dbe4e7}
*{box-sizing:border-box}body{margin:0;background:#f5f7f7;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:auto;padding:40px 28px 70px}header{background:#123541;color:#fff;border-radius:22px;padding:38px 42px}header p{max-width:750px;color:#d8e5e8;margin-bottom:0}
.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:700;color:var(--teal)}header .eyebrow{color:#8ee4ce}
h1{font-size:clamp(28px,4vw,46px);line-height:1.15;letter-spacing:-.035em;margin:16px 0}h2{font-size:26px;letter-spacing:-.025em;line-height:1.25;margin:0 0 12px}h3{font-size:20px;line-height:1.35;margin:7px 0 12px}h4{font-size:16px;margin:22px 0 8px}p{margin:8px 0 18px}
.pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:26px}.pills span{border:1px solid #456671;border-radius:30px;padding:5px 13px;font-size:13px}
.notice{border-left:3px solid var(--teal);padding:10px 18px;margin:26px 0;background:#eaf3f1;font-size:14px}.section{margin-top:42px}.lead{color:var(--muted);max-width:830px}
.grid,.comparison-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card,.report{background:#fff;border:1px solid var(--line);border-radius:14px;padding:23px}.card h3 span{display:block;font-size:13px;font-weight:400;color:var(--muted);margin-top:4px}.observation{font-size:28px;font-weight:650}.observation small{font-size:12px;font-weight:400;color:var(--muted)}svg{display:block;width:100%;height:160px;margin:12px 0}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th{font-size:11px;letter-spacing:.025em;color:var(--muted);font-weight:600}td,th{padding:10px 8px;border-bottom:1px solid var(--line);white-space:nowrap}td:first-child,th:first-child{padding-left:0}.card td,.card th{padding:8px 5px;font-size:11px}.card td:last-child{font-size:10px}
.report{margin-bottom:18px}details{border-top:1px solid var(--line);padding:13px 0 0;margin-top:13px}summary{cursor:pointer;font-weight:600;font-size:14px;padding-bottom:8px}summary:focus-visible{outline:2px solid var(--teal);outline-offset:4px}.prose{max-width:850px;padding:6px 12px}.prose h3{font-size:18px}pre{font:13px/1.7 ui-monospace,Consolas,monospace;background:#f3f6f6;border-radius:8px;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:480px;overflow:auto}
.footnote{font-size:13px;color:var(--muted)}footer{border-top:1px solid var(--line);margin-top:40px;padding-top:22px;font-size:13px;color:var(--muted)}
@media(max-width:720px){main{padding:18px 14px 40px}header{padding:28px 23px;border-radius:14px}.grid,.comparison-grid{grid-template-columns:1fr}.card,.report{padding:18px}.section{margin-top:32px}header p{font-size:14px}}
@media print{body{background:white}header{background:white;color:var(--ink);border:1px solid var(--line)}header p{color:var(--ink)}.grid{grid-template-columns:1fr 1fr}.card{break-inside:avoid}}
</style></head><body><main><header><div class="eyebrow">Undergraduate thesis · Offline portfolio demo</div>
<h1>Air-quality forecasting<br>for Thessaloniki.</h1>
<p>From hourly observations to PM10 forecasts and readable reports. Explore one saved example and see how a JSON contract changes the structure of an LLM response.</p>
<div class="pills"><span>24-hour history</span><span>+1 / +6 / +12 hours</span><span>4 test stations</span><span>No API key required</span></div></header>
<div class="notice"><strong>Historical replay — 28 January 2026, 20:00 Europe/Athens.</strong> This page displays saved predictions and cached reports. It performs no training, model inference or network requests and is not a current air-quality service.</div>
<section class="section"><h2>01 / The forecast snapshot</h2><p class="lead">Observed PM10 at the origin is shown separately from three point forecasts. Units: μg/m³. Forecasts use the recorded TCN + ANN + XGBoost ensemble.</p>
<div class="grid">''' + ''.join(cards) + '''</div><p class="footnote">Whiskers show estimated intervals at a nominal 90% level; actual test coverage is lower. The points do not describe the hours between forecast horizons. All times are hourly interval end times.</p></section>
<section class="section"><h2>02 / Three reports, two output conditions</h2><p class="lead">The original responses are preserved in Greek, the target language of the thesis application. Open a repetition to compare free text with structured JSON. Edited examples are labelled separately and were reviewed with AI-assistant assistance.</p>''' + ''.join(report_panels) + '''
<p class="footnote">The activities report uses available measured pollutants at the historical origin, including NO₂. A PM10 forecast alone is not a forecast of the overall air-quality index. Missing pollutant coverage is recorded in the supplied evidence.</p></section>
<section class="section"><h2>03 / What did the contract change?</h2><p class="lead">Three tasks × two output conditions × three repetitions = 18 saved responses. All nine contract responses matched the requested schema, making fields predictable for an application. This did not guarantee the requested length or correct interpretation.</p>
<div class="card scroll"><table><thead><tr><th>Task</th><th>Output</th><th>Schema matches</th><th>180–260 words</th><th>Mean words</th></tr></thead><tbody>''' + ''.join(comparison_rows) + '''</tbody></table></div>
<p class="footnote">Free text was not asked to be JSON, so schema conformity is not scored as a failure. These descriptive results concern one snapshot; they are not an independent user study or a general reliability estimate.</p></section>
<section class="section"><h2>04 / Recorded test performance</h2><p class="lead">Metrics below come from the complete test set for each horizon, not from the four-station snapshot. Lower RMSE and MAE are better; higher R² is better. Persistence repeats the last observed PM10 value.</p>
<details><summary>Show all models and horizons</summary><div class="card scroll"><table><thead><tr><th>Horizon</th><th>Model</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Test samples</th></tr></thead><tbody>''' + metric_rows + '''</tbody></table></div></details>
<p class="footnote">RMSE and MAE are in μg/m³. The project uses one fitted seed and a historical period previously inspected during development. High-pollution episodes and interval coverage remain limitations.</p></section>
<section class="section"><details><summary>Inspect the exact evidence supplied to the LLM</summary><pre>''' + escape(json.dumps(evidence, ensure_ascii=False, indent=2)) + '''</pre></details></section>
<footer>Local, self-contained HTML · Saved research outputs · No external scripts, fonts or services.<br>Source files, report provenance and reproduction instructions are included in the repository.</footer>
</main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="HTML output path; default: demo_output/index.html beside this script")
    parser.add_argument("--check", action="store_true", help="Verify saved fixtures without generating HTML")
    args = parser.parse_args()
    if args.check and args.output:
        parser.error("Use either --check or --output")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        bundle = load_bundle()
        if args.check:
            print("OK: fixture checksums, 4 stations, 12 forecasts, 18 original responses, 3 edited reports and 24 metric rows.")
            return
        output = (args.output or ROOT / "demo_output/index.html").resolve()
        require(output.suffix.lower() == ".html", "Output must be an .html file")
        protected = [ROOT / name for name in ("src", "configs", "examples", "results", "docs", "tests")]
        require(not any(output.is_relative_to(path) for path in protected), "Choose an output path outside repository source/fixture folders")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_html(bundle), encoding="utf-8", newline="\n")
        print("Historical replay: 28 January 2026, 20:00 Europe/Athens")
        print("Station          Observed    +1 h    +6 h   +12 h  (PM10, ug/m3)")
        for station in bundle["evidence"]["stations"]:
            values = " ".join(f'{number(f["pm10"]):7.1f}' for f in station["forecasts"])
            print(f'{ENGLISH_NAMES[station["id"]]:16} {number(station["observed_pm10"]):8.1f} {values}')
        print("\nNo model training, new inference or API call was performed.")
        print("Open in your browser:", output)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f"Demo could not run: {exc}\n")


if __name__ == "__main__":
    main()
