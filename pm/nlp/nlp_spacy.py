import logging
import re
from typing import List

import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc

from pm.nlp.base_nlp import BaseNlp

logger = logging.getLogger(__name__)

class NlpSpacy(BaseNlp):
    def __init__(self):
        self.nlp: Language | None = None

    def init_model(self):
        if self.nlp is None:
            use_cuda = True
            try:
                spacy.require_gpu()
            except:
                use_cuda = False
            try:
                self.nlp = spacy.load("en_core_web_trf")
            except Exception as e:
                logger.error(e)
                print("Do this: python -m spacy download en_core_web_trf")
                exit(1)

            try:
                from fastcoref import spacy_component
                self.nlp.add_pipe(
                    "fastcoref",
                    config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cuda:0' if use_cuda else 'cpu'}
                )
            except Exception as e:
                logger.critical(repr(e))

    def get_doc(self, text: str) -> Doc:
        self.init_model()
        return self.nlp(text)

    def resolve_coreferences(self, text: str) -> str:
        self.init_model()
        doc = self.nlp(
            text,
            component_cfg={"fastcoref": {'resolve_text': True}}
        )
        return doc._.resolved_text

    def convert_third_person_to_first_person(self, text: str, name: str) -> str:
        self.init_model()
        resolved_text = self.resolve_coreferences(text)
        lines = resolved_text.split('\n')
        converted_lines = [self._convert_line(line, name) for line in lines]
        return '\n'.join(converted_lines)

    def convert_third_person_to_instruction(self, text: str, name: str) -> str:
        self.init_model()
        resolved_text = self.resolve_coreferences(text)
        doc = self.nlp(resolved_text)

        instructions = []
        for sent in doc.sents:
            instruction = self._convert_sentence_to_instruction(sent, name)
            if instruction:
                instructions.append(instruction)

        return '\n'.join(instructions)

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts keywords from a given text using spaCy's transformer-based model,
        filtering out any keywords containing non-letter characters.

        Args:
            text (str): The text from which to extract keywords.

        Returns:
            List[str]: A list of extracted keywords.
        """
        # Process the text using the transformer model
        self.init_model()
        doc = self.nlp(text)

        # Set for keywords to avoid duplicates
        keywords = set()

        # Extract noun chunks (noun phrases)
        for chunk in doc.noun_chunks:
            lemma = chunk.lemma_.lower()
            if lemma.isalpha():  # Only add if the lemma contains letters only
                keywords.add(lemma)

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ not in ["CARDINAL", "ORDINAL", "QUANTITY"]:
                lemma = ent.lemma_.lower()
                if lemma.isalpha():  # Only add if the lemma contains letters only
                    keywords.add(lemma)

        # Additional filtering for important individual tokens
        for token in doc:
            # Filter tokens that are nouns, adjectives, or verbs (content words)
            if token.pos_ in ["NOUN", "PROPN", "ADJ", "VERB"] and not token.is_stop:
                lemma = token.lemma_.lower()
                if lemma.isalpha():  # Only add if the lemma contains letters only
                    keywords.add(lemma)

        # Return as a sorted list
        return list(keywords)

    def _convert_line(self, line, name):
        # Replace the AI's name with "I"
        line = re.sub(r'\b' + re.escape(name) + r'\b', 'I', line, flags=re.IGNORECASE)

        # Handle verb conjugation and common issues
        line = self._handle_verb_conjugation(line)

        # Handle possessive pronouns
        line = re.sub(r'\b(her|his)\b', 'my', line)
        line = re.sub(r'\b(hers|his)\b(?=\s+\w+)', 'mine', line)  # For cases like "hers is" -> "mine is"

        # Handle reflexive pronouns
        line = re.sub(r'\b(herself|himself)\b', 'myself', line)

        # Handle object pronouns
        line = re.sub(r'\b(her|him)\b', 'me', line)

        # Fix any remaining possessive issues
        line = re.sub(r'I\'s\b', 'my', line)

        return line

    def _handle_verb_conjugation(self, text):
        patterns = [
            (r'I\s+(\w+)s\b', r'I \1'),  # e.g., "I explains" -> "I explain"
            (r'I\s+is\b', 'I am'),
            (r'I\s+was\b', 'I was'),  # No change needed for past tense
            (r'I\s+has\b', 'I have'),
            (r'I\s+does\b', 'I do'),
            (r'I\s+will\b', 'I will'),  # No change needed for future tense
            (r'I\s+ha(ve|s)\s+been\b', 'I have been'),  # Fix "I ha been" -> "I have been"
            (r'I\s+i\b', 'I am'),  # Fix "I i" -> "I am"
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _convert_sentence_to_instruction(self, sent, name):
        # Check if the sentence is about the AI
        if not any(token.text.lower() == name.lower() or token.text.lower() in ['she', 'he', 'it'] for token in sent):
            return None  # Skip sentences not about the AI

        # Replace the subject with "you"
        text = sent.text
        text = re.sub(r'\b' + re.escape(name) + r'\b', 'you', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(she|he|it)\b', 'you', text, flags=re.IGNORECASE)

        # Adjust verb conjugation
        doc = self.nlp(text)
        tokens = [token for token in doc]
        for i, token in enumerate(tokens):
            if token.text.lower() == 'you' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.pos_ == 'VERB':
                    if next_token.tag_ == 'VBZ':  # 3rd person singular present
                        tokens[i + 1] = self.nlp(next_token.lemma_)[0]  # Use base form of verb
                    elif next_token.tag_ == 'VBD':  # Past tense
                        if next_token.lemma_ == 'be':
                            tokens[i + 1] = self.nlp('were')[0]
                        else:
                            tokens[i + 1] = self.nlp(next_token.text)[0]  # Keep past tense for other verbs

        # Reconstruct the sentence
        text = ' '.join([t.text for t in tokens])

        # Handle special cases
        text = re.sub(r'\byou is\b', 'you are', text)
        text = re.sub(r'\byou has\b', 'you have', text)
        text = re.sub(r'\byou does\b', 'you do', text)

        # Convert possessive pronouns
        text = re.sub(r'\b(her|his|its)\b', 'your', text)

        text = text.strip()
        tmp = text[0].upper() + text[1:]
        return tmp
