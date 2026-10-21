"""Expected service failures that leave a task outcome unverified."""


class TaskServiceError(RuntimeError):
    """A task service failed before its outcome could be verified.

    The supervisor records a failed run and keeps the application alive.
    A failed transport may follow an external effect. Never retry a world
    command or infer that nothing changed from this exception alone.
    """
