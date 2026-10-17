import yaml

def safe_load(*args, **kwargs):
    return yaml.load(*args, **kwargs, Loader=yaml.CSafeLoader)

def safe_dump(*args, **kwargs):
    return yaml.safe_dump(*args, **kwargs)