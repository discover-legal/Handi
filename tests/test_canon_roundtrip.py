"""HF state dict -> canonical -> HF is exact for every family, and the model's logits are unchanged."""
import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM  # noqa: E402

from handi.canon import family_of, from_canonical, is_parameter_buffer, to_canonical  # noqa: E402

MANIFEST = json.loads((ROOT / "results/manifest.json").read_text())


def _load(name):
    snap = MANIFEST[name]["local"]
    cfg = json.loads(Path(snap, "config.json").read_text())
    if family_of(cfg) == "qwen3_5":
        model = Qwen3_5ForCausalLM.from_pretrained(snap, dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32)
    return model.eval(), cfg


@pytest.mark.parametrize("name", ["gpt2", "tinystories-1m", "pythia-70m", "qwen3.5-0.8b"])
def test_roundtrip(name):
    model, cfg = _load(name)
    fam = family_of(cfg)
    sd = {k: v for k, v in model.state_dict().items() if not is_parameter_buffer(k)}
    canon = to_canonical(fam, sd, cfg)
    back = from_canonical(fam, canon, cfg)
    tied = model.config.tie_word_embeddings
    expected = {k for k in sd if not (tied and k == "lm_head.weight")}
    assert set(back) == expected, (set(back) ^ expected)
    for k in back:
        assert torch.equal(back[k], sd[k]), k
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        before = model(ids).logits
        model.load_state_dict(back, strict=False)
        after = model(ids).logits
    assert torch.equal(before, after)
