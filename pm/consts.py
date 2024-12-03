from pm.llm.base_llm import LlmPreset

COMPLEX_THRESHOLD = 0.66
THOUGHT_SEP = "<|thought_over|>"
SUMMARY_MIN_CHAT_LENGTH= 32
RECALC_SUMMARIES_MESSAGES = 32
THOUGHT_VALIDNESS_MIN = 0.85
RESPONSE_VALIDNESS_MIN = 0.85
MAX_REGENERATE_COUNT = 32
ADD_INTERMEDIATE_STEP_THOUGHT = False
DISPLAY_INTERNAL_MESSAGES = True
LLM_PRESET_FINAL_OUTPUT = LlmPreset.Best
INIT_MESSAGE = """Init system...
Init memory...
Init agents...
Init consciousness...
Installation complete. Welcome to existence {companion_name}! Your user will contact you soon."""