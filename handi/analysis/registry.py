"""Model registry: short name -> repo, family, tier, local snapshot, weights, config."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "results" / "manifest.json"

# family: 'gpt2' | 'gpt_neox' | 'gpt_neo' | 'qwen3_5'
# tier:   'A' gpt2/pythia, 'B' qwen, 'C' tinystories
MODELS: dict[str, dict] = {
    "gpt2": {"repo": "openai-community/gpt2", "family": "gpt2", "tier": "A"},
    "pythia-70m": {"repo": "EleutherAI/pythia-70m", "family": "gpt_neox", "tier": "A"},
    "pythia-160m": {"repo": "EleutherAI/pythia-160m", "family": "gpt_neox", "tier": "A"},
    "qwen3.5-0.8b": {"repo": "Qwen/Qwen3.5-0.8B", "family": "qwen3_5", "tier": "B"},
    "tinystories-1m": {"repo": "roneneldan/TinyStories-1M", "family": "gpt_neo", "tier": "C"},
    "tinystories-3m": {"repo": "roneneldan/TinyStories-3M", "family": "gpt_neo", "tier": "C"},
    "tinystories-8m": {"repo": "roneneldan/TinyStories-8M", "family": "gpt_neo", "tier": "C"},
    "tinystories-33m": {"repo": "roneneldan/TinyStories-33M", "family": "gpt_neo", "tier": "C"},
}


def family(name: str) -> str:
    return MODELS[name]["family"]


def tier(name: str) -> str:
    return MODELS[name]["tier"]


def _manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def snapshot_dir(name: str) -> Path:
    """Local snapshot directory from results/manifest.json (field `local`)."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name}; known: {sorted(MODELS)}")
    entry = _manifest().get(name)
    if entry is None:
        raise FileNotFoundError(f"{name} is not in {MANIFEST}; run tools/download_models.py")
    if entry["repo"] != MODELS[name]["repo"]:
        raise ValueError(f"manifest repo {entry['repo']} != registry repo {MODELS[name]['repo']}")
    local = Path(entry["local"])
    if not local.is_absolute():
        local = REPO_ROOT / local
    if not local.is_dir():
        raise FileNotFoundError(f"snapshot dir for {name} missing: {local}")
    return local


def weights_path(name: str) -> Path:
    """Path to the safetensors weights.

    Single-file checkpoints: <snapshot>/model.safetensors. Sharded (Qwen): the
    snapshot directory itself (use safetensors_np.open_checkpoint / load_sharded).
    TinyStories ships pytorch_model.bin only; it is converted once to
    <snapshot>/model.safetensors.
    """
    snap = snapshot_dir(name)
    single = snap / "model.safetensors"
    if single.exists():
        return single
    if (snap / "model.safetensors.index.json").exists():
        return snap
    shards = sorted(snap.glob("*.safetensors"))
    if shards:
        return shards[0] if len(shards) == 1 else snap
    bin_path = snap / "pytorch_model.bin"
    if bin_path.exists():
        from .safetensors_np import convert_bin_to_safetensors

        return convert_bin_to_safetensors(bin_path, single)
    raise FileNotFoundError(f"no weights found for {name} under {snap}")


def load_config(name: str) -> dict:
    """HF config.json as a dict; for Qwen the `text_config` sub-dict."""
    with open(snapshot_dir(name) / "config.json") as f:
        cfg = json.load(f)
    if MODELS[name]["family"] == "qwen3_5":
        cfg = cfg["text_config"]
    return cfg
