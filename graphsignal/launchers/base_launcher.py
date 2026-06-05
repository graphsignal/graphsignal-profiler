import logging
import os
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger('graphsignal')


class BaseLauncher(ABC):
    def __init__(self, args: List[str], enable_otel: bool = False):
        self.args: List[str] = list(args)
        # OTEL trace injection (engine --enable-trace / --otlp-traces-endpoint
        # + local collector) is opt-in via `graphsignal-run --enable-otel`.
        self.enable_otel: bool = enable_otel

    @abstractmethod
    def match(self) -> bool:
        ...

    @abstractmethod
    def launch(self) -> None:
        ...

    def executable_name(self) -> str:
        if not self.args:
            return ''
        return os.path.basename(self.args[0])
