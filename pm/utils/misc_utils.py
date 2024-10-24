def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None
