from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PRIMARY = (
    "static_quality",
    "settling_stability",
    "push_robustness",
    "articulation_quality",
    "grasp_quality",
    "actuation_quality",
)
PHYSICAL = PRIMARY[:3]
EXECUTABILITY = PRIMARY[3:]


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def percent(value):
    return None if value is None else 100.0 * value


def metric(score=None, applicable=True, strict_pass=False, diagnostics=None, reason=None):
    return {
        "score": None if score is None else max(0.0, min(1.0, float(score))),
        "applicable": bool(applicable),
        "strict_pass": bool(strict_pass),
        "diagnostics": diagnostics or {},
        "reason": reason,
    }


def _finite_applicable(row):
    if not row.get("applicable", True):
        return False
    score = row.get("score")
    return score is not None and math.isfinite(float(score))


def aggregate_trials(rows):
    applicable = [row for row in rows if _finite_applicable(row)]
    blocked = [row for row in rows if row.get("status") == "evaluation_blocked"]
    partial = [row for row in rows if row.get("status") == "partial"]
    missing = [row for row in rows if row.get("status") == "missing"]
    elapsed = [
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("elapsed_seconds") is not None
        and math.isfinite(float(row["elapsed_seconds"]))
    ]
    if not applicable:
        result = metric(
            applicable=False,
            reason=(blocked[0].get("reason") if blocked else
                    partial[0].get("reason") if partial else
                    missing[0].get("reason") if missing else "not_applicable"),
            diagnostics={
                "trial_count": len(rows),
                "blocked_trial_count": len(blocked),
                "partial_trial_count": len(partial),
                "missing_trial_count": len(missing),
                "reason_class": "evaluator" if blocked or partial or missing else "not_applicable",
                "mean_elapsed_seconds": mean(elapsed),
                "measured_trial_count": len(elapsed),
            },
        )
        if blocked:
            result["status"] = "evaluation_blocked"
            result["reason_class"] = "evaluator"
        elif partial:
            result["status"] = "partial"
            result["reason_class"] = "evaluator"
        elif missing:
            result["status"] = "missing"
            result["reason_class"] = "evaluator"
        else:
            result["status"] = "not_applicable"
            result["reason_class"] = "not_applicable"
        return result
    scores = [float(row["score"]) for row in applicable]
    passes = sum(bool(row.get("strict_pass", row.get("pass", False))) for row in applicable)
    severe = any(row.get("severe_runtime_event", False) for row in applicable)
    result = metric(
        mean(scores),
        strict_pass=passes >= math.ceil(2 * len(applicable) / 3),
        diagnostics={
            "trial_count": len(applicable),
            "trial_pass_count": passes,
            "blocked_trial_count": len(blocked),
            "partial_trial_count": len(partial),
            "missing_trial_count": len(missing),
            "severe_runtime_event": severe,
            "reason_class": "runtime" if severe else "asset",
            "mean_elapsed_seconds": mean(elapsed),
            "measured_trial_count": len(elapsed),
        },
    )
    if blocked or partial or missing:
        result["diagnostics"]["incomplete_trial_count"] = len(blocked) + len(partial) + len(missing)
    return result


def first_metric(metrics, *names):
    return next((metrics[name] for name in names if name in metrics), None)


def score_asset(inspect_row, trial_rows):
    metrics = {}
    inspect_metrics = inspect_row.get("metrics", {}) if inspect_row else {}
    physics = first_metric(inspect_metrics, "physics_configuration_quality", "asset_quality")
    collision = inspect_metrics.get("collision_quality")
    components = {
        "physics_configuration_quality": physics or metric(applicable=False, reason="missing_inspection"),
        "collision_quality": collision or metric(applicable=False, reason="missing_inspection"),
    }
    static_applicable = all(
        value.get("applicable") and value.get("score") is not None and value.get("status", "evaluated") == "evaluated"
        for value in components.values()
    )
    static_status = next(
        (
            status
            for status in ("evaluation_blocked", "partial", "missing")
            if any(value.get("status") == status for value in components.values())
        ),
        None,
    )
    static_reason = next(
        (
            value.get("reason")
            for value in components.values()
            if value.get("status") == static_status and value.get("reason")
        ),
        "static_inspection_incomplete",
    )
    static_quality = metric(
        mean(value["score"] for value in components.values()) if static_applicable else None,
        applicable=static_applicable,
        strict_pass=static_applicable and all(value.get("strict_pass", False) for value in components.values()),
        diagnostics={"components": components, "elapsed_seconds": inspect_row.get("elapsed_seconds") if inspect_row else None},
        reason=None if static_applicable else static_reason,
    )
    if static_status:
        static_quality["status"] = static_status
        static_quality["reason_class"] = "evaluator"
    elif not static_applicable:
        static_quality["status"] = "not_applicable"
        static_quality["reason_class"] = "not_applicable"
    metrics["static_quality"] = static_quality
    grouped = defaultdict(list)
    for row in trial_rows:
        trial_metrics = row.get("metrics", {})
        settling = first_metric(trial_metrics, "settling_stability", "stability")
        if settling is not None:
            grouped["settling_stability"].append(settling)
        for name in ("push_robustness", "articulation_quality", "grasp_quality", "actuation_quality"):
            if name in trial_metrics:
                grouped[name].append(trial_metrics[name])
    for name in PRIMARY[1:]:
        metrics[name] = aggregate_trials(grouped.get(name, []))
    pqs = (
        mean(metrics[name]["score"] for name in PHYSICAL)
        if all(metrics[name].get("applicable") and metrics[name].get("score") is not None for name in PHYSICAL)
        else None
    )
    eqs = (
        mean(metrics[name]["score"] for name in EXECUTABILITY)
        if all(metrics[name].get("applicable") and metrics[name].get("score") is not None for name in EXECUTABILITY)
        else None
    )
    base = inspect_row or trial_rows[0]
    return {
        "dataset": base["dataset"],
        "asset_id": base["asset_id"],
        "category": base.get("category"),
        "anchor_category": base.get("anchor_category") or (base.get("selection") or {}).get("anchor_category"),
        "anchor_robophyscan_id": base.get("anchor_robophyscan_id") or (base.get("selection") or {}).get("anchor_robophyscan_id"),
        "metrics": metrics,
        "pqs": pqs,
        "eqs": eqs,
        "pqs_strict_pass": all(metrics[name].get("strict_pass", False) for name in PHYSICAL),
        "eqs_strict_pass": eqs is not None and all(metrics[name].get("strict_pass", False) for name in EXECUTABILITY),
        "pqs_coverage": sum(metrics[name].get("applicable", False) for name in PHYSICAL) / len(PHYSICAL),
        "eqs_coverage": sum(metrics[name].get("applicable", False) for name in EXECUTABILITY) / len(EXECUTABILITY),
        "pqs_leave_one_out": {
            omitted: mean(metrics[name]["score"] for name in PHYSICAL if name != omitted and metrics[name].get("applicable"))
            for omitted in PHYSICAL
        },
    }


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - half, center + half


def quantile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] if lower == upper else values[lower] * (upper - position) + values[upper] * (position - lower)


def bootstrap_ci(values, seed=20260718, samples=2000):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return None, None
    rng = random.Random(seed)
    estimates = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples)]
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def summaries(scores):
    output = []
    datasets = sorted({row["dataset"] for row in scores})
    for dataset in datasets:
        rows = [row for row in scores if row["dataset"] == dataset]
        for name in (*PRIMARY, "pqs", "eqs"):
            entries = [row["metrics"][name] for row in rows] if name in PRIMARY else [metric(row.get(name), row.get(name) is not None, row.get(f"{name}_strict_pass", False)) for row in rows]
            applicable = [entry for entry in entries if entry.get("applicable") and entry.get("score") is not None]
            values = [entry["score"] for entry in applicable]
            passed = sum(entry.get("strict_pass", False) for entry in applicable)
            wilson_low, wilson_high = wilson(passed, len(applicable))
            bootstrap_low, bootstrap_high = bootstrap_ci(values)
            output.append({
                "dataset": dataset,
                "metric": name,
                "aggregation": "micro",
                "score_unit": "percent",
                "asset_count": len(rows),
                "applicable_count": len(applicable),
                "coverage": len(applicable) / len(rows) if rows else 0.0,
                "strict_pass_count": passed,
                "conditional_pass": passed / len(applicable) if applicable else None,
                "unconditional_success": passed / len(rows) if rows else None,
                "median": percent(quantile(values, 0.5)),
                "q1": percent(quantile(values, 0.25)),
                "q3": percent(quantile(values, 0.75)),
                "wilson_low": wilson_low,
                "wilson_high": wilson_high,
                "bootstrap_mean_low": percent(bootstrap_low),
                "bootstrap_mean_high": percent(bootstrap_high),
            })
            categories = defaultdict(list)
            for row, entry in zip(rows, entries):
                if row.get("anchor_category") and entry.get("applicable") and entry.get("score") is not None:
                    categories[row["anchor_category"]].append(entry)
            category_scores = [mean(entry["score"] for entry in group) for group in categories.values()]
            category_pass = [mean(float(entry.get("strict_pass", False)) for entry in group) for group in categories.values()]
            macro_low, macro_high = bootstrap_ci(category_scores)
            output.append({
                "dataset": dataset,
                "metric": name,
                "aggregation": "anchor_category_macro",
                "score_unit": "percent",
                "asset_count": len(rows),
                "applicable_count": sum(len(group) for group in categories.values()),
                "coverage": mean(len(group) / sum(1 for row in rows if row.get("anchor_category") == category) for category, group in categories.items()),
                "strict_pass_count": None,
                "conditional_pass": mean(category_pass),
                "unconditional_success": None,
                "median": percent(quantile(category_scores, 0.5)),
                "q1": percent(quantile(category_scores, 0.25)),
                "q3": percent(quantile(category_scores, 0.75)),
                "wilson_low": None,
                "wilson_high": None,
                "bootstrap_mean_low": percent(macro_low),
                "bootstrap_mean_high": percent(macro_high),
                "category_count": len(categories),
            })
    return output


def timing_summaries(scores):
    """Return one compact average wall-clock row per dataset and metric."""
    output = []
    for dataset in sorted({row["dataset"] for row in scores}):
        rows = [row for row in scores if row["dataset"] == dataset]
        for name in PRIMARY:
            elapsed = []
            for row in rows:
                value = row.get("metrics", {}).get(name, {})
                diagnostics = value.get("diagnostics", {})
                seconds = diagnostics.get("elapsed_seconds")
                if seconds is None:
                    seconds = diagnostics.get("mean_elapsed_seconds")
                try:
                    seconds = float(seconds)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(seconds):
                    elapsed.append(seconds)
            completed = sum(
                bool(row.get("metrics", {}).get(name, {}).get("applicable"))
                and row.get("metrics", {}).get(name, {}).get("score") is not None
                for row in rows
            )
            output.append({
                "dataset": dataset,
                "metric": name,
                "completed_count": completed,
                "measured_count": len(elapsed),
                "mean_seconds": mean(elapsed),
            })
    return output


def paired_comparison(scores):
    by_anchor = defaultdict(dict)
    for row in scores:
        if row.get("anchor_robophyscan_id"):
            by_anchor[row["anchor_robophyscan_id"]][row["dataset"]] = row
    output = []
    for metric_name in ("pqs", "eqs"):
        datasets = sorted({row["dataset"] for row in scores})
        for left_index, left in enumerate(datasets):
            for right in datasets[left_index + 1 :]:
                differences = []
                for group in by_anchor.values():
                    if left in group and right in group:
                        a, b = group[left].get(metric_name), group[right].get(metric_name)
                        if a is not None and b is not None:
                            differences.append(a - b)
                low, high = bootstrap_ci(differences)
                output.append({"metric": metric_name, "score_unit": "percentage_points", "left": left, "right": right, "paired_count": len(differences), "mean_difference": percent(mean(differences)), "bootstrap_low": percent(low), "bootstrap_high": percent(high)})
    return output


def read_jsonl(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_rows(scores):
    for row in scores:
        for name, value in row["metrics"].items():
            for key, diagnostic in value.get("diagnostics", {}).items():
                if key == "components":
                    for component_name, component in diagnostic.items():
                        yield {"dataset": row["dataset"], "asset_id": row["asset_id"], "metric": name, "secondary_metric": component_name, "diagnostic": "score", "value": component.get("score")}
                        for component_key, component_value in component.get("diagnostics", {}).items():
                            yield {"dataset": row["dataset"], "asset_id": row["asset_id"], "metric": name, "secondary_metric": component_name, "diagnostic": component_key, "value": json.dumps(component_value, ensure_ascii=False) if isinstance(component_value, (dict, list)) else component_value}
                    continue
                yield {"dataset": row["dataset"], "asset_id": row["asset_id"], "metric": name, "diagnostic": key, "value": json.dumps(diagnostic, ensure_ascii=False) if isinstance(diagnostic, (dict, list)) else diagnostic}


def failure_rows(scores):
    for row in scores:
        for name, value in row["metrics"].items():
            status = value.get("status", "evaluated" if value.get("applicable") else "not_applicable")
            if status != "evaluated" or (value.get("applicable") and not value.get("strict_pass")):
                yield {
                    "dataset": row["dataset"], "asset_id": row["asset_id"], "metric": name,
                    "status": status, "reason_class": value.get("reason_class") or value.get("diagnostics", {}).get("reason_class") or ("not_applicable" if status == "not_applicable" else "evaluator" if status in {"evaluation_blocked", "partial", "missing"} else "asset"),
                    "applicable": value.get("applicable", False), "strict_pass": value.get("strict_pass", False),
                    "reason": value.get("reason") or "strict_threshold_not_met", "score": value.get("score"),
                }
            for component_name, component in value.get("diagnostics", {}).get("components", {}).items():
                component_status = component.get("status", "evaluated" if component.get("applicable") else "not_applicable")
                if component_status != "evaluated" or (component.get("applicable") and not component.get("strict_pass")):
                    yield {
                        "dataset": row["dataset"], "asset_id": row["asset_id"], "metric": component_name, "parent_metric": name,
                        "status": component_status, "reason_class": component.get("reason_class") or component.get("diagnostics", {}).get("reason_class") or ("not_applicable" if component_status == "not_applicable" else "evaluator" if component_status in {"evaluation_blocked", "partial", "missing"} else "asset"),
                        "applicable": component.get("applicable", False), "strict_pass": component.get("strict_pass", False),
                        "reason": component.get("reason") or "strict_threshold_not_met", "score": component.get("score"),
                    }


def run_score(inspect_path, trials_path):
    inspections = {(row["dataset"], row["asset_id"]): row for row in read_jsonl(inspect_path)}
    trials = defaultdict(list)
    for row in read_jsonl(trials_path):
        trials[(row["dataset"], row["asset_id"])].append(row)
    keys = sorted(set(inspections) | set(trials))
    scores = [score_asset(inspections.get(key), trials.get(key, [])) for key in keys]
    write_jsonl(ROOT / "runs" / "scores_v5.jsonl", scores)
    primary = summaries(scores)
    primary.extend({"dataset": "paired", **row} for row in paired_comparison(scores))
    write_csv(ROOT / "reports" / "primary_metrics_v5.csv", primary)
    write_csv(ROOT / "reports" / "timing_v5.csv", timing_summaries(scores))
    write_csv(ROOT / "reports" / "diagnostics_v5.csv", diagnostic_rows(scores))
    write_csv(ROOT / "reports" / "failures_v5.csv", failure_rows(scores))
    return scores


def main():
    parser = argparse.ArgumentParser(description="Unified Executability v5 离线评分")
    parser.add_argument("--inspect", type=Path, default=ROOT / "runs" / "inspect_v5.jsonl")
    parser.add_argument("--trials", type=Path, default=ROOT / "runs" / "trials_v5.jsonl")
    args = parser.parse_args()
    run_score(args.inspect, args.trials)


if __name__ == "__main__":
    main()
