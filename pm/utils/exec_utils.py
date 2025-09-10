import inspect


def get_call_stack_str(depth_limit=None, separator="-"):
    """
    Returns a string representation of the current call stack suitable for filenames.

    Args:
        depth_limit (int, optional): Maximum number of stack frames to include.
        separator (str): Separator used between function names.

    Returns:
        str: A string of function names from the call stack.
    """
    stack = inspect.stack()
    # Skip the current function itself and possibly the one that called it
    relevant_stack = stack[1:depth_limit + 1 if depth_limit else None]

    # Reverse to show the call hierarchy from outermost to innermost
    function_names = [frame.function for frame in reversed(relevant_stack)]

    # Replace any problematic characters for filenames
    sanitized_parts = [fn.replace('<', '').replace('>', '') for fn in function_names]
    full = separator.join(sanitized_parts)
    full = full.replace(f"{separator}wrapper", "")
    idx = full.rfind(f"{separator}run")
    if idx >= 0:
        full = full[idx + 5:]
    return full[:180]