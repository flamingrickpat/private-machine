import unicodedata
from typing import Generator, Iterable, List


# ---------- helpers ----------------------------------------------------------

def is_word_char(ch: str | None) -> bool:
    """Unicode letter or digit."""
    if ch is None:
        return False
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N")

def is_space_or_punct(ch: str | None) -> bool:
    if ch is None:
        return True
    return ch.isspace() or unicodedata.category(ch)[0] in ("P", "S")

# ---------- streaming utterance extractor ------------------------------------

class UtteranceExtractor:
    def __init__(self):
        self.depth   = 0          # nesting level of "
        self.start   = None       # index where current outer quote began
        self.pending = None       # index of quote whose role will be decided next
        self.buffer  = []         # text seen so far (needed only for yield slices)

    def feed(self, text: str) -> Iterable[str]:
        """Feed a chunk and yield any *new* outer-level utterances."""
        for ch in text:

            # 1.  If there is a quote we postponed from the previous step,
            #     decide now whether it opens or closes.
            if self.pending is not None:
                prev = self.buffer[self.pending - 1] if self.pending else None
                nxt  = ch                                # current char == look-ahead

                # ------- opening vs closing decision (stream-safe) ------------
                if self.depth == 0:
                    open_q = True                        # nothing open yet
                elif is_space_or_punct(prev) and is_word_char(nxt):
                    open_q = True                        # … "Word
                elif is_word_char(prev) and is_space_or_punct(nxt):
                    open_q = False                       # word" …
                else:
                    open_q = False                       # fallback = close
                # ----------------------------------------------------------------

                if open_q:
                    if self.depth == 0:                  # new outer span
                        self.start = self.pending + 1
                    self.depth += 1
                else:                                   # closing quote
                    self.depth -= 1
                    if self.depth == 0 and self.start is not None:
                        yield "".join(self.buffer[self.start : self.pending])
                        self.start = None

                self.pending = None                      # done with that quote

            # 2.  Handle the current character itself.
            if ch == '"' and (not self.buffer or self.buffer[-1] != "\\"):
                self.pending = len(self.buffer)          # decide on next step

            self.buffer.append(ch)

    def flush(self):
        if self.pending is not None:
            prev = self.buffer[self.pending - 1] if self.pending else None
            nxt = None  # no char after EOF
            # decide open vs close exactly as inside feed
            open_q = (
                    self.depth == 0 or
                    (is_space_or_punct(prev) and is_word_char(nxt))
            )
            if not open_q:  # must be a closing quote
                self.depth -= 1
                if self.depth == 0 and self.start is not None:
                    yield "".join(self.buffer[self.start: self.pending])
        self.pending = self.start = None

    @staticmethod
    def detect_latest(text: str) -> None | str:
        lst = UtteranceExtractor.detect_all(text)
        if len(lst) > 0:
            return lst[-1]
        return None

    @staticmethod
    def detect_all(text: str) -> List[str]:
        u = UtteranceExtractor()
        res = u.feed(text)
        tmp = []
        for r in res:
            tmp.append(r)
        for r in u.flush():
            tmp.append(r)
        return tmp
