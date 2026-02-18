# Lyric Auxiliary Loss for Vocal Training

## Current State

### CN2 Has Piano Roll Auxiliary Loss ✅
```python
# From trainer_performerCN2.py

# 1. PR Head: Predicts piano roll from latents
self.pr_head = nn.Sequential(
    nn.Conv1d(C*H, 256, kernel_size=3, padding=1),
    nn.SiLU(),
    nn.Conv1d(256, 128, kernel_size=1)   # → [B, 128, T]
)

# 2. PR Loss: Binary cross-entropy
pr_logits = self.pr_head(x0_hat)  # Predict from denoised latents
pr_loss = self._pr_bce_loss(pr_logits, pr_target)

# 3. Combined into total loss
loss = recon_loss + aux_w * aux_loss + self.pr_loss_weight * pr_loss
```

**Purpose:**
- Forces latents to **encode piano roll information**
- Helps model learn musically meaningful representations
- Weight: `0.6` (significant contribution)

### Vocal Trainer Currently Has

1. ✅ **Vocal-aware PR loss** (weights boundaries more)
2. ❌ **NO dedicated lyric auxiliary loss**

---

## Should You Add Lyric Auxiliary Loss?

### Arguments FOR Adding It:

1. **Symmetry with CN2 Architecture**
   - CN2 has PR auxiliary loss
   - Vocals should have lyric auxiliary loss
   - Forces latents to encode lyric information

2. **Improved Lyric Encoding**
   - Current: Lyrics only affect latents indirectly via conditioning
   - With aux loss: Latents must **explicitly encode syllable boundaries**
   - Similar to how PR loss forces pitch encoding

3. **Better Syllable Timing**
   - Predict syllable boundaries from latents
   - Forces model to learn when syllables occur
   - Helps with pronunciation clarity

4. **Potential Loss Hierarchy**
   ```
   Total Loss = recon_loss               (Main: denoise latents)
              + aux_loss                 (Aux: classify instrument)
              + pr_loss_weight * pr_loss (Aux: encode piano roll)
              + lyric_loss_weight * lyric_loss  (NEW: encode syllables)
   ```

### Arguments AGAINST Adding It:

1. **Lyrics Already in Conditioning**
   - Lyric features already fused in conditioning encoder
   - Model sees lyrics at every step via cross-attention
   - May be redundant

2. **Different from Piano Roll**
   - Piano roll is **dense** (128 pitches × T frames)
   - Syllable boundaries are **sparse** (binary markers at T frames)
   - May not need same treatment

3. **Complexity**
   - More hyperparameters to tune
   - Another loss weight to balance
   - May slow convergence if not balanced well

4. **Your Simple Architecture**
   - You removed cross-attention for simplicity
   - Adding lyric loss adds back complexity
   - Contradicts the "keep it simple" philosophy

---

## Recommended Approach: **START WITHOUT IT**

### Why:

1. **Test Simple First**
   - Your simple conditioning is untested
   - Add complexity only if needed
   - Easier to debug without extra losses

2. **Lyric Conditioning is Strong**
   - Syllable boundaries in conditioning are frame-aligned
   - Already provides strong signal
   - May not need auxiliary loss

3. **PR Loss is Different**
   - Piano roll is the **primary musical signal**
   - Lyrics are **secondary** (guides pronunciation)
   - Not as critical to encode in latents

4. **Can Add Later**
   - Train first version without lyric loss
   - Evaluate: Are syllables well-aligned?
   - Add lyric loss only if alignment is poor

---

## If You Want to Add Lyric Loss (Optional)

Here's how to implement it properly:

### 1. Lyric Head

```python
def _init_lyric_head(self, x_latent: torch.Tensor):
    """
    Predict syllable boundaries from latents.
    Simpler than PR head (binary classification, not 128 pitches).
    """
    _, C, H, _ = x_latent.shape
    in_ch = C * H
    self.lyric_head = nn.Sequential(
        nn.Conv1d(in_ch, 128, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv1d(128, 1, kernel_size=1)   # → [B, 1, T] (binary per frame)
    ).to(device=x_latent.device, dtype=x_latent.dtype)
```

### 2. Lyric BCE Loss

```python
def _lyric_bce_loss(self, lyric_logits: torch.Tensor, lyric_target: torch.Tensor) -> torch.Tensor:
    """
    lyric_logits: [B, 1, T] - predicted syllable boundaries
    lyric_target: [B, T] - actual syllable boundaries (0/1)

    Apply positive class weighting since boundaries are sparse.
    """
    # Expand target to match logits
    lyric_target = lyric_target.unsqueeze(1)  # [B, 1, T]

    # Compute BCE with positive class weight (boundaries are rare)
    # Most frames are NOT boundaries, so weight boundaries higher
    pos_weight = torch.tensor(10.0, device=lyric_logits.device)  # Weight positive class 10x
    bce = F.binary_cross_entropy_with_logits(
        lyric_logits,
        lyric_target,
        pos_weight=pos_weight,
        reduction='none'
    )

    return bce.mean()
```

### 3. Add to Training Step

```python
# In training_step, after PR loss:

# Optional lyric BCE aux
lyric_loss = x0_hat.new_zeros(())
if self.lyric_loss_weight > 0 and vocal_conditioning is not None:
    # Check if we have syllable boundaries in this batch
    has_vocals = any(vc is not None for vc in vocal_conditioning)

    if has_vocals:
        # Initialize lyric head if needed
        if self.lyric_head is None:
            B, C, H, T_slow = x0_hat.shape
            self._init_lyric_head(x0_hat)

        # Extract syllable boundaries from batch
        lyric_target = []
        for vc in vocal_conditioning:
            if vc is not None:
                lyric_target.append(vc["syllable_boundaries"])
            else:
                # Non-vocal item - zeros
                lyric_target.append(torch.zeros(T_slow, device=x0_hat.device))

        lyric_target = torch.stack(lyric_target, dim=0)  # [B, T]

        # Predict syllable boundaries from denoised latents
        x_feat = x0_hat.reshape(B, C*H, T_slow)
        lyric_logits = self.lyric_head(x_feat)  # [B, 1, T]

        lyric_loss = self._lyric_bce_loss(lyric_logits, lyric_target)
        self.log("aux/lyric_bce", lyric_loss.detach(), on_step=True)

# Update total loss
loss = (recon_loss
        + self.adapter_l2 * cond_reg
        + aux_w * aux_loss
        + self.pr_loss_weight * pr_loss
        + self.lyric_loss_weight * lyric_loss)  # NEW
```

### 4. Add Hyperparameters

```python
# In __init__
lyric_loss_weight: float = 0.3,  # Start lower than PR (0.6)

self.lyric_loss_weight = float(lyric_loss_weight)
self.lyric_head = None

# In argparse
ap.add_argument("--lyric_loss_weight", type=float, default=0.3,
                help="Weight for syllable boundary BCE auxiliary loss")
```

---

## Comparison to Piano Roll Loss

| Aspect | Piano Roll Loss | Lyric Loss (Proposed) |
|--------|----------------|----------------------|
| **Target** | 128 pitches per frame | 1 boundary per frame |
| **Density** | Dense (many active notes) | Sparse (few boundaries) |
| **Loss Type** | Multi-label BCE | Binary BCE with pos_weight |
| **Weight** | 0.6 (high) | 0.3 (moderate, start lower) |
| **Purpose** | Encode pitch | Encode syllable timing |
| **Criticality** | High (music = pitch) | Medium (vocals = timing + pitch) |
| **Implementation** | 2-layer conv (256→128) | 2-layer conv (128→1) |

---

## Recommended Strategy

### Phase 1: Train Without Lyric Loss (Start Here) ✅

```python
# Just use vocal-aware PR loss (already implemented)
# No lyric auxiliary loss yet
lyric_loss_weight = 0.0
```

**Evaluate:**
- Are syllables well-aligned in outputs?
- Is pronunciation clear?
- Do boundaries match lyric timing?

### Phase 2: Add Lyric Loss If Needed (Only if Phase 1 fails)

```python
# Add moderate lyric loss
lyric_loss_weight = 0.3

# Monitor aux/lyric_bce during training
# Adjust weight if needed
```

**Evaluate:**
- Did syllable alignment improve?
- Is the loss weight balanced?
- Any negative effects on reconstruction?

### Phase 3: Tune Weight (If Phase 2 helps)

```python
# Experiment with weight
# Start: 0.3
# If syllables still misaligned: 0.5-0.6
# If reconstruction suffers: 0.1-0.2
```

---

## My Recommendation: **Don't Add It Yet** ⚠️

### Reasons:

1. **Your conditioning is already strong**
   - Syllable boundaries in `VocalLyricProcessor`
   - Frame-aligned features
   - Should be sufficient

2. **Keep it simple to start**
   - You just simplified from cross-attention
   - Don't add complexity back immediately
   - Test simple version first

3. **Can always add later**
   - Easy to add if syllable alignment is poor
   - Harder to debug with too many losses upfront

4. **Different from piano roll**
   - Piano roll is dense, critical signal
   - Syllables are sparse, secondary signal
   - May not benefit from same treatment

---

## Summary

| Option | Pros | Cons | Recommendation |
|--------|------|------|----------------|
| **No Lyric Loss** | Simple, fast to test, conditioning may be enough | May have poor syllable alignment | **Start here** ✅ |
| **Add Lyric Loss** | Better syllable encoding, symmetric with CN2 | More complex, slower, may be redundant | Add later if needed |

**Action Plan:**
1. ✅ Train first version **without** lyric auxiliary loss
2. ✅ Evaluate syllable alignment quality
3. ⚠️ Add lyric loss **only if** alignment is poor
4. ⚠️ Start with weight `0.3`, tune if needed

**Your simple conditioning + vocal-aware PR loss is probably enough!** Don't overcomplicate until you have evidence it's needed.

---

## Code Snippet (If You Want to Add It)

```python
# Add to trainer_performervox.py

# In __init__ (around line 231):
self.lyric_loss_weight = float(getattr(self.hparams, "lyric_loss_weight", 0.0))
self.lyric_head = None

# Add methods (after _pr_bce_loss):
def _init_lyric_head(self, x_latent: torch.Tensor):
    _, C, H, _ = x_latent.shape
    in_ch = C * H
    self.lyric_head = nn.Sequential(
        nn.Conv1d(in_ch, 128, kernel_size=3, padding=1),
        nn.SiLU(),
        nn.Conv1d(128, 1, kernel_size=1)
    ).to(device=x_latent.device, dtype=x_latent.dtype)

def _lyric_bce_loss(self, lyric_logits: torch.Tensor, lyric_target: torch.Tensor) -> torch.Tensor:
    lyric_target = lyric_target.unsqueeze(1)
    pos_weight = torch.tensor(10.0, device=lyric_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        lyric_logits, lyric_target, pos_weight=pos_weight, reduction='none'
    )
    return bce.mean()

# In training_step (after pr_loss, around line 2540):
lyric_loss = x0_hat.new_zeros(())
if self.lyric_loss_weight > 0:
    has_vocals = batch.get("vocal_conditioning") and any(
        vc is not None for vc in batch["vocal_conditioning"]
    )
    if has_vocals:
        if self.lyric_head is None:
            self._init_lyric_head(x0_hat)

        lyric_target = torch.stack([
            vc["syllable_boundaries"] if vc else torch.zeros(T_slow, device=x0_hat.device)
            for vc in batch["vocal_conditioning"]
        ])

        x_feat = x0_hat.reshape(B, C*H, T_slow)
        lyric_logits = self.lyric_head(x_feat)
        lyric_loss = self._lyric_bce_loss(lyric_logits, lyric_target)
        self.log("aux/lyric_bce", lyric_loss.detach(), on_step=True)

# Update loss (around line 2318):
loss = (recon_loss + self.adapter_l2 * cond_reg + aux_w * aux_loss
        + self.pr_loss_weight * pr_loss + self.lyric_loss_weight * lyric_loss)
```

**But again: Start with `lyric_loss_weight=0.0` and add it only if needed!**
