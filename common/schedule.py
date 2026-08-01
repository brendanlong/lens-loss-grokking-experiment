"""Log/eval cadence that cannot drift apart.

Evals used to be gated inside the periodic-logging branch, so when
``eval_every_steps`` wasn't a multiple of ``log_every_steps`` they were
silently skipped (fixed in 31817cd across three training loops). This helper
ties the two cadences together: logging always fires on eval steps, so an
eval can never be computed without being logged or skipped by misalignment.
"""


def should_log_and_eval(
    step: int,
    *,
    log_every_steps: int,
    eval_every_steps: int,
) -> tuple[bool, bool]:
    """Return ``(do_log, do_eval)`` for a 1-indexed optimizer step.

    ``do_log`` is True on either cadence. A cadence of 0 disables it.
    Because logging can now fire off the pure log cadence, window averages
    must divide by the actual number of steps since the last log, not by
    ``log_every_steps``.
    """
    do_eval = eval_every_steps > 0 and step % eval_every_steps == 0
    do_log = do_eval or (log_every_steps > 0 and step % log_every_steps == 0)
    return do_log, do_eval
