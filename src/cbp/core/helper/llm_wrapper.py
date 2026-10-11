"""Thin, non-streaming aq-llm-pivot adapter with explicit request accounting.

Importing this module never imports llmpivot. Replay experiments can therefore run
with the standard library alone. Install aq-llm-pivot only for LLM-backed runs.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from typing import Any

from cbp.core.helper.logger import Logger


class LLMWrapperError(RuntimeError):
    """A backend call failed after a known, bounded number of attempts."""

    def __init__(self, operation, original_error, attempts=0):
        self.operation = operation
        self.original_error = original_error
        self.attempts = attempts
        super().__init__(f"LLM {operation} failed after {attempts} attempt(s): "
                         f"{type(original_error).__name__}: {original_error}")


class LLMWrapper:
    """Use llm-pivot's normalized responses; own the only active retry loop.

    ``retries`` means additional attempts, not total attempts. Successful calls
    without provider usage still count as requests, but their token/cost totals
    are incomplete. Monetary amounts are estimates from explicitly supplied USD
    rates per million tokens, never an assertion about a provider invoice.
    """

    _COUNTERS = ("request_count", "attempt_count", "failed_attempt_count",
                 "usage_missing_count", "prompt_tokens", "completion_tokens", "total_tokens")

    def __init__(self, logger: Logger | None, model_id: str,
                 base_url: str | None = None, api_key: str | None = None,
                 model_type="online", api_type="chat_completions", retries=1,
                 retry_delay=1.0, retry_max_delay=30.0,
                 prompt_cost_per_million: float | None = None,
                 completion_cost_per_million: float | None = None,
                 **config_kwargs):
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
            raise ValueError("retries must be an integer between 0 and 10")
        if api_type not in ("chat_completions", "responses"):
            raise ValueError("api_type must be chat_completions or responses")
        for name, value in (("retry_delay", retry_delay), ("retry_max_delay", retry_max_delay),
                            ("prompt_cost_per_million", prompt_cost_per_million),
                            ("completion_cost_per_million", completion_cost_per_million)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                      or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (prompt_cost_per_million is None) != (completion_cost_per_million is None):
            raise ValueError("Supply both token cost rates or neither")
        if config_kwargs.get("max_retries", 0) != 0:
            raise ValueError("Use wrapper retries; llm-pivot max_retries must be 0")
        self.logger = logger or Logger(enabled=False)
        self.retries = retries
        self.retry_delay = retry_delay
        self.retry_max_delay = retry_max_delay
        self.prompt_cost_per_million = prompt_cost_per_million
        self.completion_cost_per_million = completion_cost_per_million
        config_kwargs["max_retries"] = 0
        config_kwargs.setdefault("enable_log", False)
        try:
            from llmpivot import LLMPivot, PivotConfig
            self.config = PivotConfig(model_id=model_id, base_url=base_url, api_key=api_key,
                                      model_type=model_type, api_type=api_type, **config_kwargs)
            self.llm = LLMPivot(self.config)
            # aq-llm-pivot 0.1.0 does not forward max_retries to OpenAI. Disable
            # SDK retries too, otherwise each wrapper attempt can make 3 requests.
            client = getattr(self.llm.llm, "client", None)
            if client is not None and hasattr(client, "max_retries"):
                client.max_retries = 0
        except ImportError as error:
            raise LLMWrapperError("initialize (install aq-llm-pivot for LLM mode)", error) from error
        except Exception as error:
            raise LLMWrapperError("initialize", error) from error
        self.usage = self._empty_usage()
        self.last_call_metadata: dict[str, Any] | None = None

    def _empty_usage(self):
        return {**{key: 0 for key in self._COUNTERS},
                "estimated_cost_usd": 0.0 if self.prompt_cost_per_million is not None else None,
                "by_call_type": {}}

    def reset_usage(self):
        self.usage = self._empty_usage()
        self.last_call_metadata = None

    def usage_snapshot(self):
        return copy.deepcopy(self.usage)

    def usage_delta(self, before, after=None):
        after = self.usage_snapshot() if after is None else after
        delta = {key: after.get(key, 0) - before.get(key, 0) for key in self._COUNTERS}
        old_cost, new_cost = before.get("estimated_cost_usd"), after.get("estimated_cost_usd")
        delta["estimated_cost_usd"] = new_cost - old_cost if new_cost is not None and old_cost is not None else None
        delta["by_call_type"] = {}
        for name in sorted(set(before.get("by_call_type", {})) | set(after.get("by_call_type", {}))):
            old = before.get("by_call_type", {}).get(name, self._empty_usage())
            new = after.get("by_call_type", {}).get(name, self._empty_usage())
            item = {key: new.get(key, 0) - old.get(key, 0) for key in self._COUNTERS}
            old_cost, new_cost = old.get("estimated_cost_usd"), new.get("estimated_cost_usd")
            item["estimated_cost_usd"] = new_cost - old_cost if new_cost is not None and old_cost is not None else None
            delta["by_call_type"][name] = item
        return delta

    def _increment(self, call_type, key, amount=1):
        by_type = self.usage["by_call_type"].setdefault(
            call_type, {key: value for key, value in self._empty_usage().items() if key != "by_call_type"})
        self.usage[key] += amount
        by_type[key] += amount

    def _record_usage(self, response, call_type):
        self._increment(call_type, "request_count")
        usage = response.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            self._increment(call_type, "usage_missing_count")
            return
        # Both normalized llm-pivot fields and Responses usage names are accepted.
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (prompt, completion)):
            self._increment(call_type, "usage_missing_count")
            return
        total = usage.get("total_tokens")
        if isinstance(total, bool) or not isinstance(total, int) or total < prompt + completion:
            total = prompt + completion
        for key, value in (("prompt_tokens", prompt), ("completion_tokens", completion), ("total_tokens", total)):
            self._increment(call_type, key, value)
        if self.prompt_cost_per_million is not None:
            cost = (prompt * self.prompt_cost_per_million + completion * self.completion_cost_per_million) / 1_000_000
            self._increment(call_type, "estimated_cost_usd", cost)

    @staticmethod
    def _retryable(error):
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status in (408, 409, 429) or 500 <= status < 600
        return isinstance(error, (TimeoutError, ConnectionError)) or any(
            cls.__name__ in ("APIConnectionError", "APITimeoutError", "RateLimitError")
            for cls in type(error).__mro__)

    def _generate(self, messages, call_type, **kwargs):
        kwargs["stream"] = False
        before = self.usage_snapshot()
        started = time.monotonic()
        for attempt in range(1, self.retries + 2):
            self._increment(call_type, "attempt_count")
            try:
                response = self.llm.generate(messages, **kwargs)
                if not isinstance(response, dict):
                    raise TypeError("llm-pivot must return a response dictionary")
            except Exception as error:
                self._increment(call_type, "failed_attempt_count")
                self.last_call_metadata = {"call_type": call_type, "status": "error", "attempts": attempt,
                                           "elapsed_seconds": time.monotonic() - started,
                                           "usage": self.usage_delta(before)}
                if attempt > self.retries or not self._retryable(error):
                    raise LLMWrapperError(call_type, error, attempt) from error
                delay = min(self.retry_max_delay, self.retry_delay * 2 ** (attempt - 1))
                self.logger.warning("LLM %s attempt %d failed (%s); retrying in %.2fs",
                                    call_type, attempt, type(error).__name__, delay)
                time.sleep(delay)
                continue
            self._record_usage(response, call_type)
            self.last_call_metadata = {"call_type": call_type, "status": "ok", "attempts": attempt,
                                       "elapsed_seconds": time.monotonic() - started,
                                       "model": response.get("model", self.config.model_id),
                                       "response_id": response.get("response_id"),
                                       "usage": self.usage_delta(before)}
            return response
        raise AssertionError("Unreachable retry state")

    def generate(self, messages, *, call_type="generate", **kwargs):
        return self._generate(messages, call_type, **kwargs)

    def dialogue(self, messages, *, call_type="dialogue", **kwargs):
        response = self._generate(messages, call_type, **kwargs)
        content = response.get("content")
        if content is None:
            return ""
        if not isinstance(content, str):
            raise LLMWrapperError(call_type, TypeError("Response content must be text"), 1)
        return content

    def close(self):
        client = getattr(self.llm.llm, "client", None)
        if client is not None and callable(getattr(client, "close", None)):
            client.close()
        self.llm.cleanup()
