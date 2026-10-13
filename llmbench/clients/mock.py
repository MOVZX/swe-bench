"""Mock client for validation.

mode=gold   : Returns "correct/passing" output as defined by the task grader (should be resolved).
mode=broken : Returns "failing" output (should not be resolved).

Because pass/fail output formats vary by grader (code uses FILE blocks from gold/, detection
uses gold findings, constraint/judge uses gold_answer, qa uses answer keys), actual output
generation is delegated to grader.mock_gold / mock_broken.
"""

from __future__ import annotations

from pathlib import Path

from .base import GenerationResult, LLMClient


class MockClient(LLMClient):
    def __init__(self, name: str, cfg: dict):
        super().__init__(name, cfg)
        self.mode = cfg.get("mode", "gold")
        self.current_task = None                    # Set by runner before execution
        self.current_task_dir: Path | None = None   # Backward compatibility

    def _generate(self, system: str, user: str) -> GenerationResult:
        from ..graders import get_grader

        task = self.current_task
        if task is None:
            raise RuntimeError("MockClient: current_task is not set")
        grader = get_grader(getattr(task, "grader", "code"))
        text = (
            grader.mock_broken(task) if self.mode == "broken"
            else grader.mock_gold(task)
        )
        return GenerationResult(text=text, completion_tokens=max(1, len(text) // 4))
