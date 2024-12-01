def quick_estimate_tokens(text: str, factor: float = 1.5) -> int:
    return int(len(text.split(" ")) * factor)

import re

def truncate_to_tokens(text: str, max_tokens: int, factor: float = 1.5) -> str:
    """
    Truncate text to a maximum number of tokens, clipping full sentences only.

    Args:
        text (str): The input text.
        max_tokens (int): The maximum number of tokens allowed.
        factor (float): Token estimation factor per word (default: 1.5).

    Returns:
        str: The truncated text.
    """
    # Split the text into sentences using a regex
    sentences = re.split(r'(?<=[.!?]) +', text)

    truncated_text = []
    current_tokens = 0

    for sentence in sentences:
        # Estimate the number of tokens in the current sentence
        estimated_tokens = quick_estimate_tokens(sentence, factor)
        if current_tokens + estimated_tokens > max_tokens:
            break
        truncated_text.append(sentence)
        current_tokens += estimated_tokens

    # Join the collected sentences back into a single string
    return " ".join(truncated_text)