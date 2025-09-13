import random
from typing import List, Tuple
import re
import unicodedata

from difflib import SequenceMatcher

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

def to_camel_case(snake_str):
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))


def remove_strings(text, str1, str2):
    pattern = re.escape(str1) + r"\s*" + re.escape(str2)
    return re.sub(pattern, "", text)


def replace_tagged_blocks(
        text: str,
        in_begin: str,
        in_end: str,
        out_start: str,
        out_media: str,
        out_end: str = ""
) -> Tuple[str, List[str]]:
    """
    Find every block between in_begin and in_end, collect the inner text,
    and replace each block with out_start + out_media + out_end.

    Returns (new_text, extracted_list).
    """
    pattern = re.compile(re.escape(in_begin) + r"(.*?)" + re.escape(in_end), re.DOTALL)
    extracted: List[str] = []

    def _repl(m: re.Match) -> str:
        extracted.append(m.group(1))
        return f"{out_start}{out_media}{out_end}"

    new_text = pattern.sub(_repl, text)
    return new_text, extracted


def between(s: str, a: str, b: str) -> str:
    """Return the substring strictly between the first (and only) a and b."""
    start = s.index(a) + len(a)
    end = s.index(b, start)
    return s[start:end]

# Lightweight stopwords/vendors you can tweak
STOPWORDS = {
    "the","a","an","of","and","for","with","by","on","at","to","from","in",
    "series","model","gen","edition",
    # vendor-ish prefixes often not essential for identity
    "nvidia","geforce","amd","intel","apple","microsoft","google","sony","samsung",
    "rtx","gtx"  # keep or remove depending on your domain needs
}

def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def basic_normalize(s: str) -> str:
    # Lowercase, remove accents, collapse separators, trim
    s = strip_accents(s.casefold())
    # Replace punctuation/underscores/dashes with spaces
    s = re.sub(r"[^\w\s]", " ", s)          # drop punctuation to spaces
    s = re.sub(r"[_\-]+", " ", s)           # underscores/dashes -> space
    s = re.sub(r"\s+", " ", s).strip()      # collapse whitespace
    return s

def tokenize(s: str, stopwords=STOPWORDS):
    s = basic_normalize(s)
    tokens = s.split()
    # remove trivial tokens like single letters unless numeric
    tokens = [t for t in tokens if (len(t) > 1 or t.isdigit())]
    # drop stopwords
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

def jaccard(a_tokens, b_tokens) -> float:
    A, B = set(a_tokens), set(b_tokens)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def subset_bonus(a_tokens, b_tokens) -> float:
    """If the smaller token set is fully contained in the larger, give a bonus."""
    A, B = set(a_tokens), set(b_tokens)
    if not A or not B:
        return 0.0
    small, large = (A, B) if len(A) <= len(B) else (B, A)
    return 0.92 if small.issubset(large) else 0.0

def are_similar(a: str, b: str, threshold: float = 0.82, return_score: bool = False):
    """
    Heuristic similarity:
      - exact normalized equality -> 1.0
      - token Jaccard
      - subset bonus (short is contained in long)
      - difflib ratio on normalized strings
    Returns: bool or (bool, score)
    """
    if a is None or b is None:
        return (False, 0.0) if return_score else False

    na, nb = basic_normalize(a), basic_normalize(b)
    if na == nb:
        return (True, 1.0) if return_score else True

    ta, tb = tokenize(a), tokenize(b)

    # Scores
    jac = jaccard(ta, tb)                    # token set overlap
    sub = subset_bonus(ta, tb)               # full containment bonus
    seq = SequenceMatcher(None, na, nb).ratio()  # character-level similarity

    # Simple fusion (take the best of token or char, then add a small bonus)
    base = max(jac, seq)
    score = min(1.0, base + 0.08*sub + 0.04*jac)  # tiny nudge for token agreement

    is_match = score >= threshold
    return (is_match, score) if return_score else is_match

# -----------------------------
# Examples
# -----------------------------
if __name__ == "__main__":
    tests = [
        ("Sam", "sam"),
        ("rtx 3090", "nvidia rtx 3090"),
        ("GeForce RTX 4090", "NVIDIA RTX 4090"),
        ("Sony WH-1000XM5", "WH1000XM5"),
        ("Apple iPhone 15 Pro", "iPhone 15 Pro"),
        ("Python 3.11", "python3.11"),
        ("Main Branch", "main"),
        ("Cloud Run", "Google Cloud Run"),
    ]
    for a, b in tests:
        ok, s = are_similar(a, b, return_score=True)
        print(f"{a!r} ~ {b!r} -> {ok} (score={s:.3f})")