"""Download every checkpoint the plan names, pinned to a revision, into ./models (HF_HOME).

Run: uv run python tools/download_models.py [--only name ...]
Writes results/manifest.json with repo id, revision sha, files and sizes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

from huggingface_hub import snapshot_download, HfApi  # noqa: E402

MODELS = {
    "gpt2": "openai-community/gpt2",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B",
    "tinystories-1m": "roneneldan/TinyStories-1M",
    "tinystories-3m": "roneneldan/TinyStories-3M",
    "tinystories-8m": "roneneldan/TinyStories-8M",
    "tinystories-33m": "roneneldan/TinyStories-33M",
}
ALLOW = ["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.tiktoken", "merges.txt", "vocab.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    api = HfApi()
    manifest_path = ROOT / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, repo in MODELS.items():
        if args.only and name not in args.only:
            continue
        info = api.model_info(repo, files_metadata=True)
        local = snapshot_download(repo, revision=info.sha, allow_patterns=ALLOW)
        files = {s.rfilename: s.size for s in info.siblings if s.size is not None}
        manifest[name] = {"repo": repo, "revision": info.sha, "local": local, "files": files}
        print(f"{name:16s} {repo:32s} {info.sha[:12]} {sum(files.values())/1e6:8.1f} MB  -> {local}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
