def quick_estimate_tokens(text: str, factor: float = 1.5) -> int:
    return int(len(text.split(" ")) * factor)