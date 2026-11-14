"""Turn case reports into a markdown summary or JSON."""

from __future__ import annotations

from typing import Sequence

from .core import CaseReport


def summary(reports: Sequence[CaseReport]) -> dict:
    trials = sum(len(r.trials) for r in reports)
    passed = sum(t.passed for r in reports for t in r.trials)
    return {
        "cases": len(reports),
        "trials": trials,
        "pass_rate": passed / trials if trials else 0.0,
        "cases_passing_every_trial": sum(r.pass_all for r in reports),
    }


def markdown(reports: Sequence[CaseReport]) -> str:
    s = summary(reports)
    k = len(reports[0].trials) if reports else 0
    lines = [
        f"**{s['cases_passing_every_trial']}/{s['cases']} cases passed every trial** "
        f"(pass^{k}); overall pass rate {s['pass_rate']:.0%} across {s['trials']} trials.",
        "",
        f"| Case | Pass rate | pass^{k} | Avg turns | Avg tool calls | Avg cost |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        calls = sum(len(t.trace.tool_calls) for t in r.trials) / len(r.trials)
        cost = r.mean("cost_usd")
        lines.append(f"| {r.case.name} | {r.pass_rate:.0%} | {'yes' if r.pass_all else 'no'} | "
                     f"{r.mean('turns') or 0:.1f} | {calls:.1f} | {'-' if cost is None else f'${cost:.4f}'} |")
    failing = [r for r in reports if r.failures()]
    if failing:
        lines += ["", "### Failures", ""]
        for r in failing:
            lines.append(f"**{r.case.name}**")
            for reason, n in sorted(r.failures().items(), key=lambda kv: -kv[1]):
                lines.append(f"- {n}/{len(r.trials)} trials: {reason}")
            lines.append("")
    return "\n".join(lines).rstrip()


def to_json(reports: Sequence[CaseReport]) -> dict:
    return {
        "summary": summary(reports),
        "cases": [
            {
                "name": r.case.name,
                "tags": list(r.case.tags),
                "pass_rate": r.pass_rate,
                "pass_all": r.pass_all,
                "failures": r.failures(),
                "trials": [
                    {
                        "passed": t.passed,
                        "turns": t.trace.turns,
                        "tool_calls": [c.name for c in t.trace.tool_calls],
                        "cost_usd": t.trace.cost_usd,
                        "error": t.trace.error,
                        "checks": {c.name: {"passed": res.passed, "reason": res.reason} for c, res in t.results},
                    }
                    for t in r.trials
                ],
            }
            for r in reports
        ],
    }
