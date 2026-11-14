"""A small eval harness for tool-using agents: output, trajectory and budget checks over repeated trials."""

from .core import (Case, Check, Result, ToolCall, Trace, called, called_before, contains, excludes, finished,
                   has_sections, matches, max_cost, max_tool_calls, max_turns, no_tool_errors, not_called,
                   run_case, run_suite, section_contains, section_excludes)
from .judge import llm_judge
from .report import markdown, summary, to_json

__version__ = "0.1.0"
