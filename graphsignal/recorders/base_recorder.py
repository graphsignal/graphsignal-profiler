from abc import ABC


class BaseRecorder(ABC):
    def __init__(self, pid=None, args=None):
        self.pid = pid
        self.args = args

    def setup(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def shutdown(self):
        pass

    def on_tick(self):
        pass
