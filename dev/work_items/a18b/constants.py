"""Keep test-agent context small across the configured model families.

These limits govern the independent Jesse and reviewer agents, not PM's
cognitive context. Longer evidence needs explicit scene windows. Never
silently truncate it or raise the limit to make a case pass.
"""

MAX_TEST_INPUT_TOKENS = 30_000
REVIEW_CRITERIA_PER_CALL = 1
REVIEW_REASONING_TOKENS = 768
HUMAN_THINK_SECONDS = (10.0, 22.0)
OVERLAP_TYPING_SECONDS = (2.0, 5.0)
