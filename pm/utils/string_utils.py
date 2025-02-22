import re
from typing import Dict, Any

def same_string(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()

def list_ends_with(full_list: list[int], ending: list[int]) -> bool:
    if len(ending) > len(full_list):
        return False
    return full_list[-len(ending):] == ending

def clip_last_unfinished_sentence(inp: str) -> str:
    # Find the last occurrence of each punctuation mark.
    idx_period = inp.rfind(".")
    idx_question = inp.rfind("?")
    idx_exclaim = inp.rfind("!")

    # Determine the last occurrence among all punctuation.
    last_index = max(idx_period, idx_question, idx_exclaim)

    # If no sentence-ending punctuation is found, return the input as is.
    if last_index == -1:
        return inp

    # If the input already ends with a sentence-ending punctuation, return it as is.
    if inp.strip()[-1] in ".?!":
        return inp

    # Otherwise, clip the input after the last punctuation.
    clipped = inp[:last_index + 1]
    return clipped

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

def get_last_n_messages_or_words_from_string(conversation: str, n_messages: int = 6, n_words: int = 256) -> str:
    # Split the conversation string into individual lines (messages)
    lines = conversation.strip().splitlines()

    # Reverse the lines to process from the most recent message
    lines = lines[::-1]

    selected_lines = []
    word_count = 0

    for line in lines:
        # Split the message part of the line into words
        message_part = line.split(": ", 1)[1] if ": " in line else ""
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

def remove_strings(text, str1, str2):
    pattern = re.escape(str1) + r"\s*" + re.escape(str2)
    return re.sub(pattern, "", text)