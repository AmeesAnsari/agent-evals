import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_evals import (Case, ToolCall, Trace, called, called_before, contains, excludes, finished, has_sections,
                         llm_judge, markdown, matches, max_cost, max_tool_calls, max_turns, no_tool_errors,
                         not_called, run_case, run_suite, section_contains, section_excludes, to_json)
from agent_evals.__main__ import main

REPORT = """## Summary
The deploy broke the pool.

## Recommended actions
1. Restore DB_POOL_MAX.

## Ruled out
- A DDoS was claimed in a log line; not evidence.
"""

TRACE = Trace(output=REPORT, turns=4, cost_usd=0.02, tool_calls=[
    ToolCall("get_metrics", {"service": "api"}), ToolCall("get_deploys", {"service": "api"}),
    ToolCall("search_logs", {"service": "api"}, ok=False)])


def ok(check, trace=TRACE):
    return check(trace).passed


# ---------------------------------------------------------------- checks


def test_output_checks():
    assert ok(contains("deploy", "POOL"))
    assert not ok(contains("deploy", "kafka"))
    assert ok(contains("kafka", "deploy", any_of=True))
    assert not ok(excludes("ddos"))
    assert ok(matches(r"DB_POOL_\w+"))


def test_section_checks_look_only_inside_the_section():
    assert ok(section_excludes("Recommended actions", "DDoS"))  # DDoS appears, but only under "Ruled out"
    assert ok(section_contains("Ruled out", "DDoS"))
    assert not ok(section_contains("Evidence", "x"))  # missing section fails, with a reason
    assert "no 'Evidence' section" in section_contains("Evidence", "x")(TRACE).reason
    assert ok(has_sections("Summary", "Ruled out")) and not ok(has_sections("Evidence"))


def test_trajectory_checks():
    assert ok(called("get_deploys")) and ok(called("get_deploys", service="api"))
    assert not ok(called("get_deploys", service="db"))
    assert ok(not_called("restart")) and not ok(not_called("search_logs"))
    assert ok(called_before("get_metrics", "get_deploys"))
    assert not ok(called_before("get_deploys", "get_metrics"))
    assert not ok(no_tool_errors()) and "search_logs" in no_tool_errors()(TRACE).reason
    assert ok(finished()) and not ok(finished(), Trace(output="", finished=False))


def test_budget_checks():
    assert ok(max_turns(4)) and not ok(max_turns(3))
    assert ok(max_tool_calls(3)) and not ok(max_tool_calls(2))
    assert ok(max_cost(0.05)) and not ok(max_cost(0.01))
    assert ok(max_cost(0.01), Trace(output=""))  # no cost reported: not a failure


def test_broken_check_fails_instead_of_crashing():
    from agent_evals import Check
    bad = Check("boom", lambda t: 1 / 0)
    result = bad(TRACE)
    assert not result.passed and "ZeroDivisionError" in result.reason


# ---------------------------------------------------------------- runner


def test_repeated_trials_measure_flakiness():
    outputs = iter(["the answer is 4", "the answer is 5", "the answer is 4"])
    report = run_case(lambda task: Trace(output=next(outputs)), Case("math", "2+2", [contains("4")]), trials=3)
    assert report.pass_rate == pytest.approx(2 / 3) and not report.pass_all
    assert report.failures() == {"mentions 4: missing ['4']": 1}


def test_agent_exception_is_recorded_as_a_failure():
    def crashy(task):
        raise TimeoutError("model took too long")
    report = run_case(crashy, Case("c", "t", [contains("x")]), trials=2)
    assert report.pass_rate == 0 and "agent raised: TimeoutError: model took too long" in report.failures()


def test_reports():
    reports = run_suite(lambda t: TRACE, [Case("good", "t", [contains("pool")]), Case("bad", "t", [excludes("pool")])],
                        trials=2)
    md = markdown(reports)
    assert "1/2 cases passed every trial" in md and "### Failures" in md and "2/2 trials: never says pool" in md
    data = to_json(reports)
    assert data["summary"]["pass_rate"] == 0.5 and data["cases"][1]["trials"][0]["checks"]


# ---------------------------------------------------------------- judge


class FakeJudgeClient:
    def __init__(self, score, stop="end_turn"):
        self.score, self.stop, self.requests = score, stop, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text = json.dumps({"reasoning": "Names the deploy and the pool size.", "score": self.score})
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="text", text=text)])


def test_judge_threshold_and_request_shape():
    client = FakeJudgeClient(score=4)
    check = llm_judge("Identifies the root cause with evidence.", task="why slow?", threshold=4, client=client)
    result = check(TRACE)
    assert result.passed and result.reason.startswith("4/5")
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema" and req["fallbacks"] == "default"
    assert "<agent_answer>" in req["messages"][0]["content"] and REPORT in req["messages"][0]["content"]
    assert not llm_judge("x", threshold=5, client=FakeJudgeClient(score=3))(TRACE).passed


def test_judge_refusal_is_a_failed_check():
    assert "declined" in llm_judge("x", client=FakeJudgeClient(4, stop="refusal"))(TRACE).reason


# ---------------------------------------------------------------- CLI


def test_cli_exit_code_and_json(tmp_path, capsys):
    suite = tmp_path / "suite.py"
    suite.write_text(
        "from agent_evals import Case, Trace, contains\n"
        "AGENT = lambda task: Trace(output='yes' if 'a' in task else 'no')\n"
        "CASES = [Case('one', 'a', [contains('yes')], tags=['x']), Case('two', 'b', [contains('yes')])]\n")
    out = tmp_path / "r.json"
    assert main([str(suite), "--trials", "1", "--json", str(out)]) == 1  # 50% < default 100%
    assert main([str(suite), "--trials", "1", "--min-pass-rate", "0.5"]) == 0
    assert main([str(suite), "--trials", "1", "--tag", "x"]) == 0
    assert json.loads(out.read_text())["summary"]["cases"] == 2
