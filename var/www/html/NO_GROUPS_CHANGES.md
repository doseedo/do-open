# No-Groups Pipeline Changes

## Summary
Removed instrument group/subgroup classification from the vocal training pipeline. Everything is now treated as vocals, simplifying the architecture and data structures.

## Files Modified

### 1. Manifest
**Created:** `vocal_training_manifest_READY_NOGROUPS.json`
- Removed `group` and `sub_group` fields from all 17,701 entries
- Created by: `remove_groups_from_manifest.py`

### 2. Dataloader (`dataloadvervox.py`)

#### Changes in `__getitem__`:
```python
# Old (lines ~235-240):
group = meta.get("group", "unknown")
subgroup = meta.get("sub_group", "unknown")
group_id = self.group2id.get(group, 0)
subgroup_id = self.sub2id.get(subgroup, 0)

# New:
# Everything is vocals - no group/subgroup needed
group = "vocal"
subgroup = "lead_vocal"
group_id = 0  # All vocals
subgroup_id = 0
```

#### Changes in output structure (lines ~384-386):
```python
# Old:
"instrument": {
    "group_id": torch.tensor(group_id, dtype=torch.long),
    "subgroup_id": torch.tensor(subgroup_id, dtype=torch.long),
},

# New:
# No instrument info needed - everything is vocals
"group_id": torch.tensor(0, dtype=torch.long),  # Dummy for compatibility
"subgroup_id": torch.tensor(0, dtype=torch.long),  # Dummy for compatibility
```

#### Changes in rbend masking (lines ~320-330):
```python
# Old:
# Different rbend gating for vocals vs instruments
is_vocal = ("vox" in group.lower() or "vocal" in group.lower())
if is_vocal:
    rb_mask = (rframe > 0.5).float()
else:
    rb_mask = (amp > 0.001).float()

# New:
# For vocals, rbend is gated by rframe (voiced regions only)
rb_mask = (rframe > 0.5).float()
rbend = rbend * rb_mask
```

#### Changes in `collate_latent_cond_vocal` (lines 417-418, 451-454):
```python
# Old:
group_ids.append(it["instrument"]["group_id"])
subgroup_ids.append(it["instrument"]["subgroup_id"])
...
return {
    ...
    "instrument": {
        "group_id": torch.stack(group_ids, 0),
        "subgroup_id": torch.stack(subgroup_ids, 0),
    },
    ...
}

# New:
group_ids.append(it["group_id"])
subgroup_ids.append(it["subgroup_id"])
...
return {
    ...
    "group_id": torch.stack(group_ids, 0),
    "subgroup_id": torch.stack(subgroup_ids, 0),
    ...
}
```

### 3. Trainer (`trainer_performervox.py`)

#### All conditioning encoder calls (lines 1484-1485, 1596-1597, 1708-1709, 2329-2330):
```python
# Old:
group_id=batch["instrument"]["group_id"].to(self.device),
subgroup_id=batch["instrument"]["subgroup_id"].to(self.device),

# New:
group_id=batch["group_id"].to(self.device),
subgroup_id=batch["subgroup_id"].to(self.device),
```

#### Augmentation swap logic (line 2295):
```python
# Old:
gid = batch["instrument"]["group_id"]

# New:
gid = batch["group_id"]
```

#### String masking (line 2310):
```python
# Old:
mask_major = (batch["instrument"]["group_id"] == strings_gid)

# New:
mask_major = (batch["group_id"] == strings_gid)
```

#### Loss computation (lines 2342-2343, 2786, 2793):
```python
# Old:
group_tgt = batch["instrument"]["group_id"].long().to(group_logits.device)
sub_tgt = batch["instrument"]["subgroup_id"].long().to(sub_logits.device)

# New:
group_tgt = batch["group_id"].long().to(group_logits.device)
sub_tgt = batch["subgroup_id"].long().to(sub_logits.device)
```

#### Vocal batch check (lines 2674-2678):
```python
# Old:
if "instrument" in batch and "group_id" in batch["instrument"]:
    group_ids = batch["instrument"]["group_id"]

# New:
if "group_id" in batch:
    group_ids = batch["group_id"]
```

## Testing

### Test Scripts Created:
1. **`test_vox_dataloader_simple.py`** - Tests dataloader with null path handling
2. **`test_no_groups_pipeline.py`** - Tests complete pipeline (dataloader → collate → conditioning encoder)

### Test Results:
```bash
$ conda run -n ace_step python test_no_groups_pipeline.py

✅ NO-GROUPS PIPELINE TEST PASSED!

Summary:
  Dataset size: 17,701
  Batch size: 4
  Flat structure: ✅ (no nested 'instrument' dict)
  Conditioning encoder: ✅
```

## Training Command

```bash
python trainer_performervox.py \
  --manifest_json vocal_training_manifest_READY_NOGROUPS.json \
  --checkpoint_dir /home/arlo/Data/ACE-Step/checkpoints \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --max_epochs 100 \
  --num_workers 8
```

## Key Benefits

1. **Simplified structure:** No nested `instrument` dict - flat `group_id`/`subgroup_id` fields
2. **Vocal-only logic:** Rbend masking always uses rframe (voiced regions)
3. **Consistent behavior:** All entries treated as vocals (group_id=0, subgroup_id=0)
4. **Cleaner code:** Removed conditional logic for vocals vs instruments
5. **Compatible:** Dummy group/subgroup IDs maintain compatibility with trainer's aux loss heads

## Notes

- Group/subgroup fields are set to dummy values (0) for compatibility with existing trainer code
- The trainer still has group/subgroup classification heads, but they'll just predict "vocal" for everything
- All 17,701 entries in the manifest have:
  - DCAE latents ✅
  - Piano roll ✅
  - Vocal conditioning ✅
  - Encodec/conditioning paths (null = dropout) ✅
