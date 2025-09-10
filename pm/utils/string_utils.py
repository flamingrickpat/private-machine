import random
import re

# A concise list of the most common English stop words
_common_stop_words = {
    'a', 'an', 'the', 'and', 'but', 'or', 'if', 'then', 'else', 'for', 'on', 'in',
    'with', 'to', 'from', 'by', 'of', 'at', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'so', 'that', 'this',
    'these', 'those', 'too', 'very', 'can', 'will', 'just', 'not'
}

def remove_n_words(text: str, n: int) -> str:
    """
    Removes up to n words from the input text, prioritizing common English stop words first.
    If additional removals are needed, randomly drops non-stopword tokens as a last resort.

    :param text: The original string.
    :param n: The number of words to remove.
    :return: The text with n words removed.
    """
    tokens = text.split()
    result = []
    removed_count = 0

    # First pass: drop stopwords
    non_stop_tokens = []
    for token in tokens:
        if removed_count < n and token.lower() in _common_stop_words:
            removed_count += 1
            continue
        result.append(token)
        non_stop_tokens.append(token)

    # Last resort: randomly drop non-stopwords if needed
    if removed_count < n and non_stop_tokens:
        to_remove = min(n - removed_count, len(non_stop_tokens))
        # pick random positions in the current result list to drop
        remove_indices = set(random.sample(range(len(result)), to_remove))
        result = [tok for idx, tok in enumerate(result) if idx not in remove_indices]

    return ' '.join(result)

def parse_json_from_response(text: str):
    # find the first and last curly braces
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object found in response")

    # extract the JSON substring
    json_str = text[start:end+1]
    return json_str


def third_person_to_instruction(text: str, name: str) -> str:
    """
    Convert third-person descriptions of a given person into second-person instructions.
    Returns:
        A new string with occurrences of the given name + verb (and pronoun) phrases replaced by a second-person instruction.
    """
    # Mapping of some common irregular verbs
    irregular = {
        'has': 'have',
        'does': 'do',
        'is': 'are',
        'was': 'were',
    }
    # Mapping of possessive pronouns to second-person equivalents
    pronoun_map = {
        'his': 'your',
        'her': 'your',
        'hers': 'yours',
        'their': 'your',
    }

    # Regex pattern:
    #   \b{name}\b               : match the exact name as a whole word
    #   ((?:\s+[A-Za-z]+ly)*)     : optional group of adverbs ending in 'ly'
    #   \s+([A-Za-z]+)            : the verb to convert
    #   (?:\s+(his|her|hers|their))? : optional pronoun referring to the person
    pattern = re.compile(
        rf"\b{re.escape(name)}\b((?:\s+[A-Za-z]+ly)*)\s+([A-Za-z]+)(?:\s+(his|her|hers|their))?\b"
    )

    def replacer(m: re.Match) -> str:
        adverbs = m.group(1) or ''
        verb = m.group(2)
        pronoun = m.group(3)
        low_verb = verb.lower()

        # Determine base form of the verb
        if low_verb in irregular:
            base = irregular[low_verb]
        else:
            if low_verb.endswith('ies'):
                base = low_verb[:-3] + 'y'
            elif low_verb.endswith('es'):
                base = low_verb[:-2]
            elif low_verb.endswith('s'):
                base = low_verb[:-1]
            else:
                base = low_verb

        # Start building the replacement
        parts = ["you" + adverbs, base]

        # Handle pronoun replacement if present
        if pronoun:
            pron_l = pronoun.lower()
            parts.append(pronoun_map.get(pron_l, pron_l))

        repl = ' '.join(parts)

        # Capitalize "You" if at sentence start
        start = m.start()
        if start == 0 or text[start-1] in '.!?':
            repl = repl.capitalize()
        return repl

    return pattern.sub(replacer, text)