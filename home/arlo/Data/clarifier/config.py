# /home/arlo/Data/clarifier/config.py
# Apache 2.0
# Group-specific configurations for clarifier training

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


# Full vocabulary (matches dataloader.py)
APPROVED_GROUPS = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}


@dataclass
class GroupConfig:
    """Configuration for a single instrument group clarifier."""

    name: str
    group_id: int
    subgroups: List[str]
    subgroup_ids: Dict[str, int]  # subgroup name -> local ID within group

    # Training hyperparams (can override defaults)
    hidden_dim: int = 256
    num_blocks: int = 6
    inst_emb_dim: int = 64

    # Learning rate and batch size suggestions based on data size
    suggested_lr: float = 1e-4
    suggested_batch_size: int = 8

    @property
    def num_subgroups(self) -> int:
        return len(self.subgroups)

    def subgroup_to_local_id(self, subgroup: str) -> int:
        """Convert subgroup name to local ID for this group."""
        return self.subgroup_ids.get(subgroup.lower(), 0)

    def local_id_to_subgroup(self, local_id: int) -> str:
        """Convert local ID back to subgroup name."""
        for name, idx in self.subgroup_ids.items():
            if idx == local_id:
                return name
        return "undefined"


# Pre-defined group configurations
GROUP_CONFIGS: Dict[str, GroupConfig] = {}

def _build_group_configs():
    """Build configurations for all groups."""
    global GROUP_CONFIGS

    for group_idx, group_name in enumerate(APPROVED_GROUPS):
        subgroups = APPROVED_SUBGROUPS.get(group_name, ["undefined"])
        subgroup_ids = {sg: i for i, sg in enumerate(subgroups)}

        GROUP_CONFIGS[group_name] = GroupConfig(
            name=group_name,
            group_id=group_idx,
            subgroups=subgroups,
            subgroup_ids=subgroup_ids,
        )

_build_group_configs()


# Convenience accessors
BRASS_CONFIG = GROUP_CONFIGS["brass"]
STRINGS_CONFIG = GROUP_CONFIGS["strings"]
WINDS_CONFIG = GROUP_CONFIGS["winds"]
PIANO_CONFIG = GROUP_CONFIGS["piano"]
GUITAR_CONFIG = GROUP_CONFIGS["guitar"]
BASS_CONFIG = GROUP_CONFIGS["bass"]


def get_group_config(group_name: str) -> GroupConfig:
    """Get configuration for a specific group."""
    group_name = group_name.lower()
    if group_name not in GROUP_CONFIGS:
        raise ValueError(f"Unknown group: {group_name}. Available: {list(GROUP_CONFIGS.keys())}")
    return GROUP_CONFIGS[group_name]


def save_config(config: GroupConfig, path: str):
    """Save group config to JSON."""
    data = {
        "name": config.name,
        "group_id": config.group_id,
        "subgroups": config.subgroups,
        "subgroup_ids": config.subgroup_ids,
        "hidden_dim": config.hidden_dim,
        "num_blocks": config.num_blocks,
        "inst_emb_dim": config.inst_emb_dim,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: str) -> GroupConfig:
    """Load group config from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return GroupConfig(**data)


# Print summary when run directly
if __name__ == "__main__":
    print("=== Available Group Configurations ===\n")
    for name, config in GROUP_CONFIGS.items():
        print(f"{name.upper()} (group_id={config.group_id}):")
        for sg, sg_id in config.subgroup_ids.items():
            print(f"  {sg_id}: {sg}")
        print()
