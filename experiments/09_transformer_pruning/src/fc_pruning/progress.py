from __future__ import annotations

import time


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def report_progress(
    label: str,
    current: int,
    total: int,
    started_at: float,
    *,
    every: int = 1,
    detail: str | None = None,
) -> None:
    """Print a log-friendly progress bar with elapsed time and ETA."""
    if total <= 0:
        raise ValueError("Progress total must be positive")
    current = min(max(int(current), 0), int(total))
    every = max(1, int(every))
    if current not in {1, total} and current % every:
        return

    fraction = current / total
    bar_width = 24
    complete = min(bar_width, int(round(fraction * bar_width)))
    bar = "#" * complete + "-" * (bar_width - complete)
    elapsed = time.monotonic() - started_at
    eta = elapsed * (total - current) / current if current else 0.0
    suffix = f" {detail}" if detail else ""
    print(
        f"[{label}] [{bar}] {current}/{total} ({fraction:6.2%}) "
        f"elapsed={_duration(elapsed)} eta={_duration(eta)}{suffix}",
        flush=True,
    )
