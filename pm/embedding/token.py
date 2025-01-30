def get_token(text: str) -> int:
    tokens = int(len(text.split(" ")) * 1.5)
    return tokens