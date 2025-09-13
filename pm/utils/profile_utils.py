import time
from functools import wraps

from pm.config_loader import *

logger = logging.getLogger(__name__)


# Global dictionary to store profiling data
function_stats = {}

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start

        # Store stats
        name = f"{func.__module__}.{func.__qualname__}"
        stats = function_stats.setdefault(name, {"calls": 0, "total_time": 0.0})
        stats["calls"] += 1
        stats["total_time"] += duration

        return result
    return wrapper

def print_function_stats():
    logger.info("FUNCTION_STATS")
    logger.info(f"{'Function':<60} {'Calls':<10} {'Total Time (s)':<15} {'Avg Time (s)':<15}")
    logger.info("-" * 100)
    for func_name, data in sorted(function_stats.items(), key=lambda item: item[1]['total_time'], reverse=True):
        calls = data["calls"]
        total_time = data["total_time"]
        avg_time = total_time / calls if calls else 0
        logger.info(f"{func_name:<60} {calls:<10} {total_time:<15.6f} {avg_time:<15.6f}")