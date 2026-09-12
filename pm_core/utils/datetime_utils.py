import datetime


def get_causal_datetime() -> datetime.datetime:
    """
    Return real datetime now in UTC, or get from test infrastructure for dry runs.
    For when simulating past (not implemented yet)
    """
    return datetime.datetime.utcnow()

def get_real_datetime() -> datetime.datetime:
    """
    Real world time, for system.
    """
    return datetime.datetime.utcnow()