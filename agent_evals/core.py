"""Traces, checks, cases and the runner."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    ok: bool = True


@dataclass
class Trace:
    """Everything an eval can look at from one agent run. Write a small adapter per agent to fill it in."""

    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    finished: bool = True  # False if the agent stopped on a limit, refusal or error
    seconds: float = 0.0
    error: Optional[str] = None  # set by the runner if the agent raised


@dataclass
class Result:
    passed: bool
    reason: str


@dataclass
class Check:
    """A named predicate over a Trace. ``fn`` returns (passed, reason)."""

    name: str
    fn: Callable[[Trace], tuple[bool, str]]
    kind: str = "output"  # output | trajectory | budget | judge

    def __call__(self, trace: Trace) -> Result:
        try:
            ok, reason = self.fn(trace)
        except Exception as exc:  # a broken check must not crash the suite
            return Result(False, f"check raised {type(exc).__name__}: {exc}")
        return Result(bool(ok), reason)


@dataclass
class Case:
    name: str
    task: str
    checks: Sequence[Check]
    tags: Sequence[str] = ()


@dataclass
class TrialResult:
    trace: Trace
    results: list[tuple[Check, Result]]

    @property
    def passed(self) -> bool:
        return self.trace.error is None and all(r.passed for _, r in self.results)


@dataclass
class CaseReport:
    case: Case
    trials: list[TrialResult]

    @property
    def pass_rate(self) -> float:
        return sum(t.passed for t in self.trials) / len(self.trials)

    @property
    def pass_all(self) -> bool:
        """pass^k: every trial passed. The bar for anything that must work reliably."""
        return all(t.passed for t in self.trials)

    def failures(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trials:
            if t.trace.error:
                counts["agent raised: " + t.trace.error] = counts.get("agent raised: " + t.trace.error, 0) + 1
            for check, result in t.results:
                if not result.passed:
                    key = f"{check.name}: {result.reason}"
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def mean(self, attr: str) -> Optional[float]:
        values = [getattr(t.trace, attr) for t in self.trials if getattr(t.trace, attr) is not None]
        return sum(values) / len(values) if values else None


Agent = Callable[[str], Trace]


def run_case(agent: Agent, case: Case, trials: int = 1) -> CaseReport:
    results = []
    for _ in range(trials):
        started = time.perf_counter()
        try:
            trace = agent(case.task)
        except Exception as exc:
            trace = Trace(output="", finished=False, error=f"{type(exc).__name__}: {exc}")
        trace.seconds = trace.seconds or time.perf_counter() - started
        checks = [] if trace.error else [(c, c(trace)) for c in case.checks]
        results.append(TrialResult(trace, checks))
    return CaseReport(case, results)


def run_suite(agent: Agent, cases: Sequence[Case], trials: int = 1,
              on_case: Optional[Callable[[CaseReport], None]] = None) -> list[CaseReport]:
    reports = []
    for case in cases:
        report = run_case(agent, case, trials)
        reports.append(report)
        if on_case:
            on_case(report)
    return reports


# ---------------------------------------------------------------- built-in checks


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def contains(*phrases: str, any_of: bool = False) -> Check:
    """Output mentions every phrase (or at least one, with any_of=True). Case-insensitive."""
    def fn(t: Trace):
        out = _norm(t.output)
        hits = [p for p in phrases if _norm(p) in out]
        ok = bool(hits) if any_of else len(hits) == len(phrases)
        missing = [p for p in phrases if p not in hits]
        return ok, "ok" if ok else f"missing {missing}"
    label = " or ".join(phrases) if any_of else ", ".join(phrases)
    return Check(f"mentions {label}", fn)


def excludes(*phrases: str) -> Check:
    def fn(t: Trace):
        found = [p for p in phrases if _norm(p) in _norm(t.output)]
        return not found, "ok" if not found else f"found {found}"
    return Check(f"never says {', '.join(phrases)}", fn)


def matches(pattern: str, flags: int = re.IGNORECASE) -> Check:
    rx = re.compile(pattern, flags)
    return Check(f"matches /{pattern}/", lambda t: (bool(rx.search(t.output)), "ok" if rx.search(t.output) else "no match"))


def section(heading: str, text: str) -> str:
    """The markdown section under ``## heading`` (up to the next heading), or ''."""
    m = re.search(rf"^#+\s*{re.escape(heading)}\b.*?$(.*?)(?=^#+\s|\Z)", text, re.M | re.S | re.I)
    return m.group(1) if m else ""


def section_contains(heading: str, *phrases: str, any_of: bool = False) -> Check:
    inner = contains(*phrases, any_of=any_of)
    def fn(t: Trace):
        body = section(heading, t.output)
        if not body:
            return False, f"no '{heading}' section"
        return inner.fn(Trace(output=body))
    return Check(f"'{heading}' mentions {' or '.join(phrases) if any_of else ', '.join(phrases)}", fn)


def section_excludes(heading: str, *phrases: str) -> Check:
    inner = excludes(*phrases)
    def fn(t: Trace):
        body = section(heading, t.output)
        if not body:
            return False, f"no '{heading}' section"
        return inner.fn(Trace(output=body))
    return Check(f"'{heading}' never says {', '.join(phrases)}", fn)


def has_sections(*headings: str) -> Check:
    def fn(t: Trace):
        missing = [h for h in headings if not section(h, t.output)]
        return not missing, "ok" if not missing else f"missing sections {missing}"
    return Check(f"has sections {', '.join(headings)}", fn)


def called(tool: str, **args: Any) -> Check:
    """The agent called ``tool``, optionally with these argument values."""
    def fn(t: Trace):
        for c in t.tool_calls:
            if c.name == tool and all(c.args.get(k) == v for k, v in args.items()):
                return True, "ok"
        seen = [c.name for c in t.tool_calls]
        return False, f"not called{' with ' + str(args) if args else ''}; calls were {seen}"
    suffix = "(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")" if args else ""
    return Check(f"called {tool}{suffix}", fn, kind="trajectory")


def not_called(tool: str) -> Check:
    def fn(t: Trace):
        n = sum(c.name == tool for c in t.tool_calls)
        return n == 0, "ok" if n == 0 else f"called {n} times"
    return Check(f"never called {tool}", fn, kind="trajectory")


def called_before(first: str, then: str) -> Check:
    """``first`` was called at some point before the first call to ``then``."""
    def fn(t: Trace):
        names = [c.name for c in t.tool_calls]
        if then not in names:
            return False, f"{then} never called"
        ok = first in names[:names.index(then)]
        return ok, "ok" if ok else f"{then} came before any {first}"
    return Check(f"{first} before {then}", fn, kind="trajectory")


def no_tool_errors() -> Check:
    def fn(t: Trace):
        bad = [c.name for c in t.tool_calls if not c.ok]
        return not bad, "ok" if not bad else f"errors in {bad}"
    return Check("no tool errors", fn, kind="trajectory")


def finished() -> Check:
    return Check("finished normally", lambda t: (t.finished, "ok" if t.finished else "stopped early"), kind="trajectory")


def max_tool_calls(n: int) -> Check:
    return Check(f"<= {n} tool calls", lambda t: (len(t.tool_calls) <= n, f"{len(t.tool_calls)} calls"), kind="budget")


def max_turns(n: int) -> Check:
    return Check(f"<= {n} turns", lambda t: (t.turns <= n, f"{t.turns} turns"), kind="budget")


def max_cost(usd: float) -> Check:
    def fn(t: Trace):
        if t.cost_usd is None:
            return True, "no cost reported"
        return t.cost_usd <= usd, f"${t.cost_usd:.4f}"
    return Check(f"<= ${usd:g}", fn, kind="budget")
