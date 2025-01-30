import re


def convert_to_list(input_text):
    """
    Converts a text containing numbers, dashes, or line breaks into a clean list of strings.
    """
    # Remove any leading/trailing whitespace
    input_text = input_text.strip()

    # Regex pattern to split on line breaks, numbers, or dashes
    # Matches any of:
    # 1. Numbers followed by a period or a parenthesis (e.g., "1.", "2)", "3.")
    # 2. Dashes as list indicators (e.g., "- item")
    # 3. Line breaks
    # 4. Any unnecessary whitespace around list items
    items = re.split(r'(?:^\d+[\.\)]|^\-\s|\n)+', input_text, flags=re.MULTILINE)

    # Remove empty items and strip whitespace
    clean_items = [item.strip() for item in items if item.strip()]

    return clean_items