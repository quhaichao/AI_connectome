from .config import paper_config, smoke_config


def run_comparison(*args, **kwargs):
    # Lazy import keeps configuration/analysis utilities inspectable on login nodes
    # that do not expose the training environment.
    from .train import run_comparison as _run_comparison

    return _run_comparison(*args, **kwargs)

__all__ = ["paper_config", "smoke_config", "run_comparison"]
