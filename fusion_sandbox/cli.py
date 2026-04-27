from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import validate_config_file
from .pipeline import (
    allan_variance_file,
    analyze_fault_file,
    analyze_residual_file,
    compare_experiment,
    export_c_header,
    export_matlab_reference,
    list_example_configs,
    report_run,
    plan_experiment_file,
    run_experiment,
    summarize_run,
    sweep_tuning,
    validate_dataset,
)


def _print(data: dict) -> None:
    print(json.dumps(data, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fusion_sandbox", description="Navigation state-estimation and sensor-fusion sandbox.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Validate and normalize a JSON configuration.")
    validate.add_argument("--config", required=True)

    examples = sub.add_parser("list-examples", help="List bundled example configurations and validation status.")
    examples.add_argument("--dir", default="examples/configs")

    run = sub.add_parser("run", help="Run a configured experiment.")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)

    compare = sub.add_parser("compare", help="Run filter or scenario comparisons.")
    compare.add_argument("--config", required=True)
    compare.add_argument("--out", required=True)

    sweep = sub.add_parser("sweep-tuning", help="Run all built-in tuning profiles for one scenario.")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--out", required=True)

    summarize = sub.add_parser("summarize", help="Summarize an existing summary JSON.")
    summarize.add_argument("--input", required=True)

    export = sub.add_parser("export-matlab", help="Export MATLAB-friendly reference data.")
    export.add_argument("--config", required=True)
    export.add_argument("--out", required=True)

    report = sub.add_parser("report", help="Write or refresh an HTML report for a run directory.")
    report.add_argument("--run", required=True)

    dataset = sub.add_parser("validate-dataset", help="Validate measurement CSV or JSONL event datasets.")
    dataset.add_argument("--input", required=True)
    dataset.add_argument("--jsonl-out")

    residuals = sub.add_parser("analyze-residuals", help="Estimate residual noise and whiteness from an innovations CSV.")
    residuals.add_argument("--input", required=True)
    residuals.add_argument("--out")

    allan = sub.add_parser("allan-variance", help="Compute Allan deviation for a numeric measurement column.")
    allan.add_argument("--input", required=True)
    allan.add_argument("--column", required=True)
    allan.add_argument("--sample-rate-hz", type=float, required=True)
    allan.add_argument("--out")

    faults = sub.add_parser("analyze-faults", help="Build a sensor-health and fault timeline from an innovations CSV.")
    faults.add_argument("--input", required=True)
    faults.add_argument("--out")

    plan = sub.add_parser("plan-experiment", help="List planned scenarios and reproducibility hashes for a config.")
    plan.add_argument("--config", required=True)
    plan.add_argument("--out")

    c_header = sub.add_parser("export-c-header", help="Export run CSV traces as C header reference arrays.")
    c_header.add_argument("--run", required=True)
    c_header.add_argument("--out", required=True)
    c_header.add_argument("--max-rows", type=int)

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config, errors = validate_config_file(args.config)
        if errors:
            _print({"valid": False, "errors": errors})
            raise SystemExit(2)
        _print({"valid": True, "name": config.get("name"), "filters": config["estimator"]["filters"], "profiles": config["estimator"]["profiles"]})
    elif args.command == "list-examples":
        _print(list_example_configs(args.dir))
    elif args.command == "run":
        _print(run_experiment(args.config, args.out))
    elif args.command == "compare":
        _print(compare_experiment(args.config, args.out))
    elif args.command == "sweep-tuning":
        _print(sweep_tuning(args.config, args.out))
    elif args.command == "summarize":
        _print(summarize_run(Path(args.input)))
    elif args.command == "export-matlab":
        _print(export_matlab_reference(args.config, args.out))
    elif args.command == "report":
        _print(report_run(args.run))
    elif args.command == "validate-dataset":
        _print(validate_dataset(args.input, args.jsonl_out))
    elif args.command == "analyze-residuals":
        _print(analyze_residual_file(args.input, args.out))
    elif args.command == "allan-variance":
        _print(allan_variance_file(args.input, args.column, args.sample_rate_hz, args.out))
    elif args.command == "analyze-faults":
        _print(analyze_fault_file(args.input, args.out))
    elif args.command == "plan-experiment":
        _print(plan_experiment_file(args.config, args.out))
    elif args.command == "export-c-header":
        _print(export_c_header(args.run, args.out, args.max_rows))


if __name__ == "__main__":
    main()
