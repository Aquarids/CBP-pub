"""Explicit LLM configuration; replay code needs neither .env nor credentials."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from cbp.core.helper.llm_wrapper import LLMWrapper
from cbp.core.helper.logger import Logger


class LLMSettings:
    """Environment wins over an optionally, explicitly selected dotenv file.

    This intentionally supports only literal KEY=VALUE lines, optional ``export``
    and paired quotes. It does not execute shell expressions or expand variables.
    Missing explicit files fail rather than silently selecting another endpoint.
    """

    def __init__(self, path: str | Path | None = None, *, environ: Mapping[str, str] | None = None):
        values: dict[str, str] = {}
        if path is not None:
            for number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    raise ValueError(f"Invalid dotenv assignment on line {number}")
                key, value = (part.strip() for part in line.split("=", 1))
                if not key:
                    raise ValueError(f"Empty dotenv key on line {number}")
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key] = value
        self.values = {**values, **(os.environ if environ is None else environ)}

    def _required(self, name):
        value = self.values.get(name, "").strip()
        if not value:
            raise ValueError(f"Missing LLM setting: {name}")
        return value

    def wrapper(self, logger: Logger | None = None, *, prefix="LLM_") -> LLMWrapper:
        """Build an OpenAI-compatible llm-pivot backend from explicit settings.

        LOCAL_LLM_ uses the same API protocol for a local server; model_type=local
        is an opt-in for llm-pivot's Ollama backend, not an inference from a URL.
        """
        effort = self.values.get(prefix + "MODEL_EFFORT")
        prompt_cost = self.values.get(prefix + "PROMPT_COST_PER_MILLION")
        completion_cost = self.values.get(prefix + "COMPLETION_COST_PER_MILLION")
        return LLMWrapper(
            logger, model_id=self._required(prefix + "MODEL_ID"),
            base_url=self._required(prefix + "BASE_URL"),
            api_key=self.values.get(prefix + "API_KEY"),
            model_type=self.values.get(prefix + "MODEL_TYPE", "online"),
            api_type=self.values.get(prefix + "API_STYLE", "chat_completions"),
            retries=int(self.values.get(prefix + "RETRIES", "1")),
            retry_delay=float(self.values.get(prefix + "RETRY_DELAY", "1")),
            retry_max_delay=float(self.values.get(prefix + "RETRY_MAX_DELAY", "30")),
            timeout=int(self.values.get(prefix + "TIMEOUT", "60")),
            prompt_cost_per_million=float(prompt_cost) if prompt_cost is not None else None,
            completion_cost_per_million=float(completion_cost) if completion_cost is not None else None,
            # llm-pivot maps reasoning_effort to Responses' reasoning object.
            llm_default_params={"reasoning_effort": effort} if effort else {},
            max_retries=0, enable_log=False,
        )

    def cloud_wrapper(self, logger: Logger | None = None) -> LLMWrapper:
        return self.wrapper(logger)

    def local_wrapper(self, logger: Logger | None = None) -> LLMWrapper:
        return self.wrapper(logger, prefix="LOCAL_LLM_")
