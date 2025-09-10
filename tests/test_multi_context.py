import contextlib
import datetime

import llama_cpp
from llama_cpp import Llama

llm = Llama(model_path=r"C:\workspace\AI\gemma\qat\gemma-3-12b-it-q4_0.gguf", n_gpu_layers=-1, n_ctx=14000, flash_attn_type=1)

# --- monkeypatches -------------------------------------------------

def mark(self) -> int:
    """Return current KV position (number of tokens already in cache for seq 0)."""
    return self.n_tokens  # this mirrors how Llama.eval tracks position

def rollback_to(self, pos: int, seq_id: int = 1):
    """
    Drop KV tokens in [pos, inf) for the given sequence, without copying.
    Also fix Python-side counters so the next eval() appends from pos.
    """
    # remove KV cells at positions >= pos
    self.n_tokens = pos
    bt = self.tokenize(" ".encode("utf-8"), add_bos=False, special=False)
    self.eval(bt)

def fork_seq(self, dst_seq_id: int, src_seq_id: int = 0, p0: int = 0, p1: int | None = None):
    """
    Assign KV range [p0, p1) from src_seq_id to dst_seq_id (no memory copy).
    Handy to branch generations or run parallel beams that share a prefix.
    """
    if p1 is None:
        p1 = self._ctx.kv_cache_seq_pos_max(src_seq_id) + 1
    self._ctx.kv_cache_seq_cp(src_seq_id, dst_seq_id, p0, p1)

# attach to the instance
llm.mark = mark.__get__(llm, Llama)
llm.rollback_to = rollback_to.__get__(llm, Llama)
llm.fork_seq = fork_seq.__get__(llm, Llama)


big_prompt = """<start_of_turn>system
You are an expert story writer.
<start_of_turn>user
Write a fantasy story.
<start_of_turn>assistant
For generations wizards kept their knowledge to themselves, practicing their magical arts far from those who should never wield such power. They are convinced that forbidden esoteric knowledge should remain forbidden at all cost. However, in 1450, a wizard peering into the future foresees a world where knowledge is not forbidden, and it spreads like wildfire throughout the land. It all starts with an inventor named Johannes Gutenberg, who builds a machine called the printing press, allowing books to be published and distributed on a massive scale. A renaissance then overtakes the world, with knowledge of every form being distributed amongst the people. However, with the power to enlighten the masses, the wizard-seer fears it is only a matter of time before everyone knows the magical secrets that he and his brethren possess. So, with a snap of his finger, the wizard kills Gutenberg and all those who would dare use machines to distribute knowledge. Soon thereafter, Europe falls into a second dark age, and wizards become the puppet masters of kings and emperors throughout the world, ensuring that their knowledge remains theirs alone."""


def _detokenize(tokens, special: bool = False) -> str:
    def silent_decode(b: bytes) -> str:
        return_val = ""
        with contextlib.suppress(UnicodeDecodeError):
            return_val = b.decode('utf-8')
        return return_val

    if not isinstance(tokens, list):
        tokens = [tokens]
    return silent_decode(llm.tokenizer()._model.detokenize(tokens, special=special))


t = datetime.datetime.now()
base_tokens = llm.tokenize(big_prompt.encode("utf-8"), add_bos=True, special=True)
llm.eval(base_tokens)
base_pos = llm.mark()
lst = []
for j in range(128):
    token = llm.sample(idx=None)
    lst.append(token)
    llm.eval([token])
    # check for exit
    if llama_cpp.llama_token_is_eog(llm._model.vocab, token):
        break

cur_completion_as_text = _detokenize(lst, special=False)
tmp = cur_completion_as_text.replace("\n", "")[:64]
print(tmp)
print(datetime.datetime.now() - t)

for i in range(100):
    t = datetime.datetime.now()
    llm.rollback_to(base_pos, seq_id=1)
    lst = []
    for j in range(128):
        token = llm.sample(idx=None)
        lst.append(token)
        llm.eval([token])
        # check for exit
        if llama_cpp.llama_token_is_eog(llm._model.vocab, token):
            break

    cur_completion_as_text = _detokenize(lst, special=False)
    tmp = cur_completion_as_text.replace("\n", "")[:64]
    print(tmp)
    print(datetime.datetime.now() - t)
