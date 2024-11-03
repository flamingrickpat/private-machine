import re
from typing import List


def generate_instructions_from_feedback(reasons: List[str]) -> str:
    instructions = []

    for reason in reasons:
        # Check for common pitfalls and create a concise instruction
        if "overly verbose" in reason or "excessive length" in reason:
            instructions.append("Keep responses concise and to the point.")

        if "irrelevant content" in reason or "does not directly address" in reason:
            instructions.append("Focus solely on answering the user's question directly.")

        if "introspective thoughts" in reason:
            instructions.append("Avoid introspective language; stay factual and relevant.")

        if "philosophical conversation" in reason:
            instructions.append("Avoid philosophical or abstract commentary.")

        if "difficult for the user to get clear information" in reason:
            instructions.append("Ensure the response is clear and easy to understand.")

    # Remove duplicates and join instructions into a single prompt-friendly string
    unique_instructions = set(instructions)
    return " ".join(unique_instructions)
