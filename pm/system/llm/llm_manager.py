from typing import Dict, Any

class LlmManager:
    def __init__(self, model_map: Dict[str, Any], test_mode: bool=False):
        self.test_mode = test_mode
        self.model_map = model_map