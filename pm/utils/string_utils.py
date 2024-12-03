import re
from typing import Dict, Any

def same_string(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()

def list_ends_with(full_list: list[int], ending: list[int]) -> bool:
    if len(ending) > len(full_list):
        return False
    return full_list[-len(ending):] == ending

def truncate_after_last_period(long_string):
    last_dot_index = long_string.rfind('.')
    if last_dot_index == -1:  # No dot found in the string
        return long_string
    return long_string[:last_dot_index + 1]


def find_python_functions_in_text(text, functions):
    regex_pattern = "|".join(f'{func}\(.*?\)' for func in functions)
    return re.findall(regex_pattern, text)

def join_memory_paths(*args):
    paths = []
    for arg in args:
        paths.append(arg.strip("/"))
    return "/".join(paths)


class StringConstantMeta(type):
    def __getattr__(cls, name):
        if name in cls._values:
            return cls._values[name]
        raise AttributeError(f"{cls.__name__} has no attribute '{name}'")

    def __iter__(cls):
        return iter(cls._values.values())

    def __contains__(cls, item):
        return item in cls._values.values()

class StringConstant(metaclass=StringConstantMeta):
    _values: Dict[str, str] = {}

    @classmethod
    def add(cls, name: str, value: str = None):
        if value is None:
            value = name
        cls._values[name] = value
        return value

    def __class_getitem__(cls, item):
        if isinstance(item, str):
            return cls.add(item)
        elif isinstance(item, tuple) and len(item) == 2:
            return cls.add(*item)
        raise ValueError("Invalid item format")


def get_last_n_messages_or_words_from_string(conversation: str, n_messages: int = 6, n_tokens: int = 256) -> str:
    # Split the conversation string into individual lines, ensuring we consider ":" in the first 20 characters
    lines = []
    current_message = []

    n_words = n_tokens / 1.5

    for line in conversation.strip().splitlines():
        # Check if there's a ":" in the first 20 characters of the line
        if ":" in line[:32]:
            # If we already have collected parts of a previous message, save it
            if current_message:
                lines.append("\n".join(current_message))
                current_message = []
        current_message.append(line)

    # Append the last message if there are leftover lines
    if current_message:
        lines.append("\n".join(current_message))

    # Reverse the lines to process from the most recent message
    lines = lines[::-1]

    selected_lines = []
    word_count = 0

    for line in lines:
        # Split the message part of the line into words
        message_part = line.split(": ", 1)[1] if ": " in line[:20] else ""
        words_in_message = message_part.split()
        message_word_count = len(words_in_message)

        # Check if adding this message exceeds the word limit
        if word_count + message_word_count > n_words and len(selected_lines) >= n_messages:
            break

        # Update the word count and add the line
        word_count += message_word_count
        selected_lines.append(line)

        # Stop if we reach the message count limit
        if len(selected_lines) >= n_messages:
            break

    # Reverse again to keep the order from oldest to newest in the selection
    selected_lines = selected_lines[::-1]

    # Join the selected lines into the final text block format
    message_block = "\n".join(selected_lines)
    return message_block

def get_text_after_keyword(text: str, keyword: str) -> str:
    # Find the position of the keyword in the text
    keyword_position = text.find(keyword)

    # If the keyword is found, return the text after it
    if keyword_position != -1:
        return text[keyword_position + len(keyword):].strip()

    # If the keyword is not found, return the whole string
    return text


def clip_to_sentence(text, max_chars):
    """
    Clips a long string to a full sentence within the specified number of characters.

    Parameters:
        text (str): The input string.
        max_chars (int): The maximum number of characters allowed in the clipped string.

    Returns:
        str: The clipped string ending at a full sentence.
    """
    if len(text) <= max_chars:
        return text

    # Find all sentences using a regex
    sentences = re.split(r'(?<=[.!?]) +', text)

    clipped_text = ""
    for sentence in sentences:
        if len(clipped_text) + len(sentence) <= max_chars:
            clipped_text += sentence + " "
        else:
            break

    # Strip trailing spaces
    return clipped_text.strip()


def clip_to_sentence_by_words(text, max_words):
    """
    Clips a long string to a full sentence within the specified number of words.

    Parameters:
        text (str): The input string.
        max_words (int): The maximum number of words allowed in the clipped string.

    Returns:
        str: The clipped string ending at a full sentence.
    """
    # Split the input text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    clipped_text = []
    word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        if word_count + len(sentence_words) <= max_words:
            clipped_text.append(sentence)
            word_count += len(sentence_words)
        else:
            break

    return " ".join(clipped_text).strip()
