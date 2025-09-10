import math


def get_token_count(data: str, char_to_token_ratio: float = 3.5) -> int:
    """Estimates token count for a knoxel's content."""
    if not data:
        return 1  # Minimal count for structure
    if not isinstance(data, str):
        try:
            data = data.get_story_element()
        except:
            data = str(data)
    # A common rough estimate is 3-4 characters per token.
    return math.ceil(len(data) / char_to_token_ratio)
