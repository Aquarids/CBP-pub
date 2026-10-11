"""Small Decoy-style logging facade without configuring the root logger."""
from __future__ import annotations

import logging
from pathlib import Path

ENV_LOCAL = "local"
ENV_DEV = "dev"
ENV_PROD = "prod"


class Logger:
    def __init__(self, name="cbp", env=ENV_LOCAL, enabled=True,
                 level=None, log_file=None):
        self.env = env
        self.enabled = enabled
        # An instance-owned logger prevents one experiment changing another's handlers.
        self.logger = logging.Logger(name)
        self.logger.propagate = False
        self.logger.setLevel(level or (logging.DEBUG if self.is_debug() else logging.INFO))
        if not enabled:
            self.logger.addHandler(logging.NullHandler())
            return
        handler = logging.FileHandler(Path(log_file), encoding="utf-8") if log_file else logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
        self.logger.addHandler(handler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def is_debug(self):
        return self.env in (ENV_LOCAL, ENV_DEV)

    def debug(self, message, *args, **kwargs):
        if self.enabled:
            self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        if self.enabled:
            self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        if self.enabled:
            self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        if self.enabled:
            self.logger.error(message, *args, **kwargs)

    def log_exception(self, error):
        if self.enabled:
            self.logger.error("%s", error, exc_info=(type(error), error, error.__traceback__))

    def close(self):
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
