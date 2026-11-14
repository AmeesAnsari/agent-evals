"""Run a suite file: ``python -m agent_evals SUITE.py [--trials N] [--min-pass-rate 0.9] [--json out.json]``.

A suite file defines ``AGENT`` (a callable task -> Trace) and ``CASES`` (a list of Case).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .core import run_suite
from .report import markdown, summary, to_json


def load_suite(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    for name in ("AGENT", "CASES"):
        if not hasattr(module, name):
            raise SystemExit(f"{path} must define {name}")
    return module.AGENT, module.CASES


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agent_evals", description="Run an agent eval suite.")
    p.add_argument("suite", type=Path)
    p.add_argument("--trials", type=int, default=3, help="runs per case (agents are not deterministic)")
    p.add_argument("--tag", help="only run cases with this tag")
    p.add_argument("--min-pass-rate", type=float, default=1.0,
                   help="exit non-zero below this overall pass rate (for CI)")
    p.add_argument("--json", type=Path, help="also write a JSON report here")
    a = p.parse_args(argv)

    agent, cases = load_suite(a.suite)
    if a.tag:
        cases = [c for c in cases if a.tag in c.tags]

    def progress(report):
        mark = "PASS" if report.pass_all else "FAIL"
        print(f"  {mark}  {report.case.name}  ({report.pass_rate:.0%})", file=sys.stderr)

    print(f"Running {len(cases)} cases x {a.trials} trials from {a.suite}", file=sys.stderr)
    reports = run_suite(agent, cases, a.trials, on_case=progress)
    print("\n" + markdown(reports))
    if a.json:
        a.json.write_text(json.dumps(to_json(reports), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if summary(reports)["pass_rate"] >= a.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
