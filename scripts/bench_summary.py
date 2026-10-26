"""Summarize one scripts/bench-baseline.sh result directory into summary.md.

Inputs: environment.json, runs.tsv, records/*.json (tuckmill bench record
schema v4), optional llama-bench.json and mlx.json. Every number written is
labeled locally-calibrated (measured here, uncontrolled for thermals) and is
never a claim. Warm statistics exclude the first repetition of each bench
invocation (it includes worker spawn and model load); runs flagged
failed/discarded never enter a median.
"""

import csv
import json
import statistics
import sys
from pathlib import Path


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=1, unit=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def warm_samples(record):
    samples = record.get("samples", [])
    return samples[1:] if len(samples) > 1 else []


def run_stats(record):
    warm = warm_samples(record)
    cold = record.get("samples", [])[:1]
    return {
        "ttft_warm_ms": median([s.get("ttft_ms") for s in warm]),
        # End-to-end: model tokens generated over total request time
        # (TTFT included), the record schema v4 definition.
        "e2e_warm_tps": median(
            [s["generated_tokens"] * 1000.0 / s["total_ms"] for s in warm if s.get("total_ms")]
        ),
        "decode_warm_tps": median([s.get("throughput_decode_tps") for s in warm]),
        "ttft_cold_load_ms": cold[0].get("ttft_ms") if cold else None,
        "footprint_peak_bytes": record.get("summary", {}).get("footprint_peak_bytes"),
        "warm_samples": len(warm),
        "engine": record.get("engine", {}),
    }


def main() -> int:
    out = Path(sys.argv[1])
    env = json.loads((out / "environment.json").read_text())
    rows = list(csv.DictReader((out / "runs.tsv").open(), delimiter="\t"))

    per_engine = {}
    per_round = {}
    lines = []
    for row in rows:
        stats = None
        if row["status"] == "ok":
            record = json.loads((out / "records" / row["record"]).read_text())
            rate = record.get("summary", {}).get("completion_rate", 0.0)
            if rate < 1.0:
                # The harness records failed samples with their error in
                # notes; an incomplete record never enters a median.
                notes = [n for s in record.get("samples", []) for n in s.get("notes", [])]
                failed = next((n for n in notes if n.startswith("failed:")), "incomplete")
                row = dict(row, status=f"failed ({rate:.0%} complete) - {failed[:120]}")
            else:
                stats = run_stats(record)
                per_engine.setdefault(row["engine"], []).append(stats)
                per_round.setdefault(row["round"], {})[row["engine"]] = stats
        lines.append((row, stats))

    md = []
    md.append(f"# Baseline: {env['artifact']} on {env['chip']} ({env['tier']})")
    md.append("")
    md.append(
        "Label: **locally-calibrated** (measured on this machine; thermals and "
        "background load uncontrolled). Not a claim; see `docs/claims.md`."
    )
    md.append("")
    md.append(f"- macOS {env['macos']}, power: {env['power_source']}, memory {env['memory_bytes']} B")
    md.append(f"- tuckmill `{env['tuckmill_commit'][:12]}` (dirty tree: {env['tuckmill_tree_dirty']}), llama.cpp {env['llama_cpp']}")
    md.append(f"- artifact `{env['artifact']}`, {env['artifact_bytes']} B, sha256 `{env['artifact_sha256']}`")
    shape = env["shape"]
    md.append(
        f"- shape: {shape['input']} in / {shape['output']} out, {shape['repetitions']} repetitions per "
        f"invocation, context {shape['context']}, {shape['rounds']} ABBA rounds; frozen synthetic "
        "token ids (standard-v1), identical for both harness engines"
    )
    md.append("- warm = repetitions 2..n of an invocation; cold-load = repetition 1 (includes worker spawn + model load; page cache uncontrolled)")
    md.append("")
    md.append("## Runs (ABBA order)")
    md.append("")
    md.append("| round | slot | engine | status | reclaimable before | swap delta | warm TTFT | warm e2e tok/s | warm decode tok/s | cold-load TTFT | footprint peak |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row, stats in lines:
        s = stats or {}
        fp = s.get("footprint_peak_bytes")
        md.append(
            f"| {row['round']} | {row['slot']} | {row['engine']} | {row['status']} | "
            f"{row['reclaimable_mib_before']} MiB | {row['swap_delta_mib']} MiB | "
            f"{fmt(s.get('ttft_warm_ms'), 0, ' ms')} | {fmt(s.get('e2e_warm_tps'), 2)} | "
            f"{fmt(s.get('decode_warm_tps'), 2)} | {fmt(s.get('ttft_cold_load_ms'), 0, ' ms')} | "
            f"{fmt(fp / 1048576 if fp else None, 0, ' MiB')} |"
        )
    md.append("")
    md.append("## Medians across valid rounds")
    md.append("")
    md.append("| engine | rounds | warm TTFT | warm e2e tok/s | warm decode tok/s | footprint peak |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for engine, runs in sorted(per_engine.items()):
        fp = median([r["footprint_peak_bytes"] for r in runs])
        md.append(
            f"| {engine} | {len(runs)} | {fmt(median([r['ttft_warm_ms'] for r in runs]), 0, ' ms')} | "
            f"{fmt(median([r['e2e_warm_tps'] for r in runs]), 2)} | "
            f"{fmt(median([r['decode_warm_tps'] for r in runs]), 2)} | "
            f"{fmt(fp / 1048576 if fp else None, 0, ' MiB')} |"
        )
    md.append("")

    ratios_decode, ratios_ttft = [], []
    for pair in per_round.values():
        a, b = pair.get("tuckcore-metal"), pair.get("llama-metal")
        if a and b and a["decode_warm_tps"] and b["decode_warm_tps"]:
            ratios_decode.append(a["decode_warm_tps"] / b["decode_warm_tps"])
        if a and b and a["ttft_warm_ms"] and b["ttft_warm_ms"]:
            ratios_ttft.append(a["ttft_warm_ms"] / b["ttft_warm_ms"])
    md.append("## Paired ratios (TuckCore / llama.cpp, same round)")
    md.append("")
    md.append(
        f"- warm decode tok/s ratio: median {fmt(median(ratios_decode), 3)} over {len(ratios_decode)} paired rounds "
        f"({', '.join(f'{r:.3f}' for r in ratios_decode) or 'none'})"
    )
    md.append(
        f"- warm TTFT ratio (higher = slower prefill): median {fmt(median(ratios_ttft), 2)} over {len(ratios_ttft)} paired rounds"
    )
    md.append("")

    md.append("## Vendor tool: pinned llama-bench (own prompt tokens; not identity-matched)")
    md.append("")
    lb = out / "llama-bench.json"
    if lb.exists() and lb.stat().st_size > 0:
        try:
            entries = json.loads(lb.read_text())
            md.append("| test | tokens/s (avg ± stddev) | backend |")
            md.append("| --- | --- | --- |")
            for e in entries:
                test = f"pp{e['n_prompt']}" if e.get("n_prompt") else f"tg{e['n_gen']}"
                md.append(f"| {test} | {e['avg_ts']:.2f} ± {e['stddev_ts']:.2f} | {e.get('backends', '')} |")
        except (ValueError, KeyError) as exc:
            md.append(f"llama-bench output unreadable: {exc}")
        stderr = out / "llama-bench.stderr"
        if stderr.exists():
            for line in stderr.read_text().splitlines():
                if "peak memory footprint" in line or "maximum resident set size" in line:
                    md.append(f"- `/usr/bin/time -l`: {line.strip()}")
    else:
        md.append("not run")
    md.append("")

    md.append("## MLX")
    md.append("")
    mlx = out / "mlx.json"
    if mlx.exists() and mlx.stat().st_size > 0:
        probe = json.loads(mlx.read_text())
        versions = f"mlx {probe.get('mlx', '?')}, mlx-lm {probe.get('mlx_lm', '?')}; " if "mlx" in probe else ""
        md.append(f"available: {probe['available']} — {versions}{probe.get('reason', '')}")
    else:
        md.append("probe produced no output (see mlx.stderr)")
    md.append("")

    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
