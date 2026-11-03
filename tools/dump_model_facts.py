"""Write docs/models/<name>.md: config and tensor list (name, shape, dtype) for every downloaded checkpoint."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))
from safetensors import safe_open
import torch

manifest = json.loads((ROOT / "results/manifest.json").read_text())
out_dir = ROOT / "docs/models"; out_dir.mkdir(parents=True, exist_ok=True)
for name, m in manifest.items():
    snap = Path(m["local"])
    cfg = json.loads((snap / "config.json").read_text())
    lines = [f"# {name}  ({m['repo']} @ {m['revision'][:12]})", "", "## config.json", "```json", json.dumps(cfg, indent=1), "```", "", "## tensors", "", "| name | shape | dtype |", "|---|---|---|"]
    st = sorted(snap.glob("*.safetensors"))
    n = 0
    if st:
        for p in st:
            with safe_open(str(p), "pt") as f:
                for k in f.keys():
                    s = f.get_slice(k); lines.append(f"| {k} | {list(s.get_shape())} | {s.get_dtype()} |"); n += int(torch.tensor(s.get_shape()).prod()) if s.get_shape() else 1
    else:
        sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
        for k, v in sd.items():
            lines.append(f"| {k} | {list(v.shape)} | {str(v.dtype).replace('torch.', '')} |"); n += v.numel()
    lines += ["", f"total elements: {n:,}", f"source: {'safetensors' if st else 'pytorch_model.bin'}"]
    (out_dir / f"{name}.md").write_text("\n".join(lines))
    print(f"{name:16s} {n:>14,} elements  {'safetensors' if st else 'bin'}")
