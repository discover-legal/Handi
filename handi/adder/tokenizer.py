"""13-symbol tokenizer for the adder: digits 0-9, '+', '=', BOS."""
from __future__ import annotations

DIGITS = [str(i) for i in range(10)]
PLUS, EQ, BOS = 10, 11, 12
VOCAB = DIGITS + ["+", "=", "<bos>"]
N_DIGITS = 10          # operand width (10-digit numbers, zero padded)
N_SUM = N_DIGITS + 1   # sum width (11 digits, leading overflow digit)
N_CTX = 1 + N_DIGITS + 1 + N_DIGITS + 1 + N_SUM  # 34


def encode_prompt(a: int, b: int) -> list[int]:
    if not (0 <= a < 10 ** N_DIGITS and 0 <= b < 10 ** N_DIGITS):
        raise ValueError("operands must have at most 10 digits")
    sa, sb = f"{a:0{N_DIGITS}d}", f"{b:0{N_DIGITS}d}"
    return [BOS] + [int(c) for c in sa] + [PLUS] + [int(c) for c in sb] + [EQ]


def decode_sum(ids: list[int]) -> int:
    if len(ids) != N_SUM or any(not 0 <= t <= 9 for t in ids):
        raise ValueError(f"not an {N_SUM}-digit answer: {ids}")
    return int("".join(str(t) for t in ids))


def to_text(ids: list[int]) -> str:
    return " ".join(VOCAB[t] for t in ids)
