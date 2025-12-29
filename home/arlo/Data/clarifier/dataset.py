# /home/arlo/Data/clarifier/dataset.py
# Apache 2.0
# Dataset for loading synthetic-real latent pairs

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


# Vocab (same as dataloader.py)
APPROVED_GROUPS = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}


def _fix_path(path: str, path_replace: Optional[tuple] = None) -> str:
    """Apply path prefix replacement if specified."""
    if path_replace and len(path_replace) == 2:
        return path.replace(path_replace[0], path_replace[1])
    return path


def _safe_pt_load(path: Optional[str], path_replace: Optional[tuple] = None) -> Optional[torch.Tensor]:
    """Safely load a .pt file."""
    try:
        if not path:
            return None
        path = _fix_path(path, path_replace)
        p = Path(path)
        if not (p.exists() and p.is_file()):
            return None
        obj = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for k in ("synthetic", "real", "latent", "latents", "data"):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    return obj[k]
        return None
    except Exception:
        return None


def _pad_or_crop(tensor: torch.Tensor, target_len: int, dim: int = -1) -> torch.Tensor:
    """Pad or crop tensor along specified dimension."""
    cur_len = tensor.shape[dim]
    if cur_len == target_len:
        return tensor
    if cur_len > target_len:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_len)
        return tensor[tuple(slices)]
    # Pad
    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_len - cur_len
    pad = tensor.new_zeros(*pad_shape)
    return torch.cat([tensor, pad], dim=dim)


class ClarifierPairDataset(Dataset):
    """
    Dataset for loading pre-generated synthetic-real latent pairs.

    Expected pair file format (.pt):
    {
        'synthetic': [8, 16, T] or [B, 8, 16, T],
        'real': [8, 16, T] or [B, 8, 16, T],
        'group_id': int or tensor,
        'subgroup_id': int or tensor,
    }

    For group-specific training, use filter_group to only load pairs from
    a single instrument group (e.g., "brass").
    """

    def __init__(
        self,
        pairs_dir: str,
        window_size: int = 512,
        random_crop: bool = True,
        require_alignment: bool = True,
        seed: Optional[int] = None,
        filter_group: Optional[str] = None,  # e.g., "brass", "strings"
        group_config: Optional[Any] = None,   # GroupConfig for local ID mapping
    ):
        super().__init__()
        self.pairs_dir = Path(pairs_dir)
        self.window_size = window_size
        self.random_crop = random_crop
        self.require_alignment = require_alignment
        self.filter_group = filter_group.lower() if filter_group else None
        self.group_config = group_config

        self.rng = np.random.default_rng(seed)

        # Build group/subgroup lookups
        self.group2id = {g: i for i, g in enumerate(APPROVED_GROUPS)}
        self.id2group = {i: g for g, i in self.group2id.items()}
        all_subs = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
        self.sub2id = {sg: i for i, sg in enumerate(all_subs)}
        self.id2sub = {i: sg for sg, i in self.sub2id.items()}

        # Find all pair files
        self.pair_files: List[Path] = []
        if self.pairs_dir.exists():
            all_files = sorted(self.pairs_dir.glob("*.pt"))

            if self.filter_group is not None:
                # Filter to only include files from the target group
                target_group_id = self.group2id.get(self.filter_group)
                if target_group_id is None:
                    raise ValueError(f"Unknown group: {self.filter_group}. "
                                     f"Available: {list(self.group2id.keys())}")

                print(f"[ClarifierPairDataset] Filtering for group '{self.filter_group}' (id={target_group_id})")
                for pf in all_files:
                    try:
                        data = torch.load(pf, map_location="cpu", weights_only=False)
                        gid = data.get("group_id", -1)
                        if isinstance(gid, torch.Tensor):
                            gid = int(gid.item())
                        if gid == target_group_id:
                            self.pair_files.append(pf)
                    except Exception:
                        continue
                print(f"[ClarifierPairDataset] Found {len(self.pair_files)}/{len(all_files)} "
                      f"pairs for group '{self.filter_group}'")
            else:
                self.pair_files = all_files
                print(f"[ClarifierPairDataset] Found {len(self.pair_files)} pair files in {pairs_dir}")

    def __len__(self) -> int:
        return len(self.pair_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_path = self.pair_files[idx]

        try:
            data = torch.load(pair_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ClarifierPairDataset] Failed to load {pair_path}: {e}")
            # Return dummy data
            return self._dummy_item()

        synthetic = data.get("synthetic")
        real = data.get("real")

        if synthetic is None or real is None:
            return self._dummy_item()

        # Handle batch dimension if present
        if synthetic.dim() == 4 and synthetic.shape[0] == 1:
            synthetic = synthetic.squeeze(0)
        if real.dim() == 4 and real.shape[0] == 1:
            real = real.squeeze(0)

        # Ensure [8, 16, T] format
        if synthetic.dim() != 3 or synthetic.shape[0] != 8 or synthetic.shape[1] != 16:
            return self._dummy_item()
        if real.dim() != 3 or real.shape[0] != 8 or real.shape[1] != 16:
            return self._dummy_item()

        T_syn = synthetic.shape[-1]
        T_real = real.shape[-1]

        # Align lengths
        T_min = min(T_syn, T_real)
        synthetic = synthetic[..., :T_min]
        real = real[..., :T_min]

        # Random or center crop to window size
        T = synthetic.shape[-1]
        if T > self.window_size:
            if self.random_crop:
                start = int(self.rng.integers(0, T - self.window_size + 1))
            else:
                start = (T - self.window_size) // 2
            end = start + self.window_size
            synthetic = synthetic[..., start:end]
            real = real[..., start:end]
        elif T < self.window_size:
            # Pad to window size
            synthetic = _pad_or_crop(synthetic, self.window_size, dim=-1)
            real = _pad_or_crop(real, self.window_size, dim=-1)

        # Get instrument IDs
        group_id = data.get("group_id", 0)
        subgroup_id = data.get("subgroup_id", 0)
        if isinstance(group_id, torch.Tensor):
            group_id = int(group_id.item())
        if isinstance(subgroup_id, torch.Tensor):
            subgroup_id = int(subgroup_id.item())

        # Convert to local subgroup ID if using group_config
        local_subgroup_id = subgroup_id
        if self.group_config is not None:
            # Map global subgroup name to local ID within the group
            subgroup_name = self.id2sub.get(subgroup_id, "undefined")
            local_subgroup_id = self.group_config.subgroup_to_local_id(subgroup_name)

        return {
            "synthetic": synthetic.float(),
            "real": real.float(),
            "group_id": torch.tensor(group_id, dtype=torch.long),
            "subgroup_id": torch.tensor(subgroup_id, dtype=torch.long),
            "local_subgroup_id": torch.tensor(local_subgroup_id, dtype=torch.long),
            "meta": {"path": str(pair_path)},
        }

    def _dummy_item(self) -> Dict[str, Any]:
        """Return dummy item for failed loads."""
        return {
            "synthetic": torch.zeros(8, 16, self.window_size),
            "real": torch.zeros(8, 16, self.window_size),
            "group_id": torch.tensor(0, dtype=torch.long),
            "subgroup_id": torch.tensor(0, dtype=torch.long),
            "local_subgroup_id": torch.tensor(0, dtype=torch.long),
            "meta": {"path": "dummy"},
        }


class ClarifierManifestDataset(Dataset):
    """
    Dataset that loads synthetic and real latents on-the-fly from manifest.
    Use this when pairs haven't been pre-generated.

    Manifest format (JSON list):
    [
        {
            "synthetic_path": "/path/to/synthetic.pt",
            "latent_path": "/path/to/real_latent.pt",
            "group": "brass",
            "sub_group": "trumpet",
            ...
        },
        ...
    ]
    """

    def __init__(
        self,
        manifest_path: str,
        window_size: int = 512,
        random_crop: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.window_size = window_size
        self.random_crop = random_crop
        self.rng = np.random.default_rng(seed)

        # Build group/subgroup lookups
        self.group2id = {g: i for i, g in enumerate(APPROVED_GROUPS)}
        all_subs = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})
        self.sub2id = {sg: i for i, sg in enumerate(all_subs)}

        # Load manifest
        with open(manifest_path, 'r') as f:
            self.manifest: List[Dict[str, Any]] = json.load(f)

        # Filter to only items with both synthetic and real
        self.manifest = [
            item for item in self.manifest
            if item.get("synthetic_path") and item.get("latent_path")
        ]

        print(f"[ClarifierManifestDataset] Loaded {len(self.manifest)} items from {manifest_path}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self.manifest[idx]

        # Load latents
        synthetic = _safe_pt_load(meta.get("synthetic_path"))
        real = _safe_pt_load(meta.get("latent_path"))

        if synthetic is None or real is None:
            return self._dummy_item()

        # Handle batch dimension
        if synthetic.dim() == 4:
            synthetic = synthetic.squeeze(0)
        if real.dim() == 4:
            real = real.squeeze(0)

        # Ensure correct shape
        if synthetic.shape[0] != 8 or synthetic.shape[1] != 16:
            return self._dummy_item()
        if real.shape[0] != 8 or real.shape[1] != 16:
            return self._dummy_item()

        # Align and crop
        T_min = min(synthetic.shape[-1], real.shape[-1])
        synthetic = synthetic[..., :T_min]
        real = real[..., :T_min]

        T = synthetic.shape[-1]
        if T > self.window_size:
            if self.random_crop:
                start = int(self.rng.integers(0, T - self.window_size + 1))
            else:
                start = (T - self.window_size) // 2
            end = start + self.window_size
            synthetic = synthetic[..., start:end]
            real = real[..., start:end]
        elif T < self.window_size:
            synthetic = _pad_or_crop(synthetic, self.window_size, dim=-1)
            real = _pad_or_crop(real, self.window_size, dim=-1)

        # Get instrument IDs
        group = (meta.get("group") or "undefined").lower()
        subgroup = (meta.get("sub_group") or "undefined").lower()
        group_id = self.group2id.get(group, 0)
        subgroup_id = self.sub2id.get(subgroup, 0)

        return {
            "synthetic": synthetic.float(),
            "real": real.float(),
            "group_id": torch.tensor(group_id, dtype=torch.long),
            "subgroup_id": torch.tensor(subgroup_id, dtype=torch.long),
            "meta": {
                "synthetic_path": meta.get("synthetic_path", ""),
                "real_path": meta.get("latent_path", ""),
            },
        }

    def _dummy_item(self) -> Dict[str, Any]:
        return {
            "synthetic": torch.zeros(8, 16, self.window_size),
            "real": torch.zeros(8, 16, self.window_size),
            "group_id": torch.tensor(0, dtype=torch.long),
            "subgroup_id": torch.tensor(0, dtype=torch.long),
            "meta": {"path": "dummy"},
        }


def collate_clarifier(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for clarifier datasets."""
    max_T = max(item["synthetic"].shape[-1] for item in batch)

    synthetics = []
    reals = []
    group_ids = []
    subgroup_ids = []
    local_subgroup_ids = []
    metas = []

    for item in batch:
        synthetics.append(_pad_or_crop(item["synthetic"], max_T, dim=-1))
        reals.append(_pad_or_crop(item["real"], max_T, dim=-1))
        group_ids.append(item["group_id"])
        subgroup_ids.append(item["subgroup_id"])
        # Handle local_subgroup_id if present
        if "local_subgroup_id" in item:
            local_subgroup_ids.append(item["local_subgroup_id"])
        metas.append(item["meta"])

    result = {
        "synthetic": torch.stack(synthetics, dim=0),
        "real": torch.stack(reals, dim=0),
        "group_id": torch.stack(group_ids, dim=0),
        "subgroup_id": torch.stack(subgroup_ids, dim=0),
        "meta": metas,
    }

    if local_subgroup_ids:
        result["local_subgroup_id"] = torch.stack(local_subgroup_ids, dim=0)

    return result


if __name__ == "__main__":
    # Quick test with dummy data
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy pair file
        pair_path = os.path.join(tmpdir, "test_pair_0.pt")
        torch.save({
            "synthetic": torch.randn(8, 16, 256),
            "real": torch.randn(8, 16, 256),
            "group_id": 4,  # brass
            "subgroup_id": 5,  # trumpet
        }, pair_path)

        # Test dataset
        ds = ClarifierPairDataset(tmpdir, window_size=128, seed=0)
        print(f"Dataset length: {len(ds)}")

        item = ds[0]
        print(f"Synthetic shape: {item['synthetic'].shape}")
        print(f"Real shape: {item['real'].shape}")
        print(f"Group ID: {item['group_id']}")
        print(f"Subgroup ID: {item['subgroup_id']}")

        # Test collate
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_clarifier)
        for batch in loader:
            print(f"Batch synthetic: {batch['synthetic'].shape}")
            print(f"Batch real: {batch['real'].shape}")
            break
