"""Stub for generation trajectory logger - used for debugging only."""

class GenerationLogger:
    """Stub logger class."""
    def __init__(self, *args, **kwargs):
        pass

    def log_generation(self, *args, **kwargs):
        return {}

def get_logger(*args, **kwargs):
    """Return a stub logger."""
    return GenerationLogger()
