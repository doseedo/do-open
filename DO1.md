# DO1 Training Script — Code Agent Instructions

## OVERVIEW

Build the complete training pipeline for DO1, a 3.3B parameter DiT (Diffusion Transformer) that operates in DCAE latent space via flow matching. The model takes three latent tensors as input and produces one latent tensor as output. Everything is latent — no MIDI, no text, no structured conditioning in the model path.

**Reference architecture:** ACE-Step's DiT backbone. DO1 modifies the input layer and adds cross-attention.

---

## PART 1: MODEL ARCHITECTURE

### 1.1 Core Architecture

DO1 is a DiT (Diffusion Transformer) with these modifications:

```
Inputs:
  x_noisy [B, 8, 16, T]   — being denoised via flow matching
  x_cond  [B, 8, 16, T]   — primary reference (optionally corrupted)
  x_ref   [B, 8, 16, T']  — secondary reference (style/timbre), or zeros
  mask    [B, 1, 16, T]    — 1=preserve x_cond content, 0=generate fresh
  t       [B]              — flow matching timestep

Output:
  v_pred  [B, 8, 16, T]   — predicted velocity field
```

### 1.2 PatchEmbed (17 channels)

Concatenate x_noisy (8ch) + x_cond (8ch) + mask (1ch) = 17 channels along channel dim.

```python
# Concatenate along channel dimension
x_input = torch.cat([x_noisy, x_cond, mask.expand(-1, -1, -1, -1)], dim=1)  # [B, 17, 16, T]

# Patch embed: treat each (freq_patch, time_patch) as a token
# Suggested patch size: (4, 4) on the [16, T] spatial dims
# This gives 4 freq patches × (T/4) time patches = T tokens per sample
# Each patch: 17 channels × 4 freq × 4 time = 272-dim → project to model_dim

patch_embed = nn.Conv2d(17, model_dim, kernel_size=(4, 4), stride=(4, 4))
# Input: [B, 17, 16, T] → Output: [B, model_dim, 4, T//4]
# Reshape to: [B, 4 * T//4, model_dim] = [B, num_tokens, model_dim]
```

### 1.3 x_ref Cross-Attention

x_ref is separately patch-embedded (8 channels, same patch size) into its own token sequence. DO1's transformer layers cross-attend to these tokens.

```python
ref_patch_embed = nn.Conv2d(8, model_dim, kernel_size=(4, 4), stride=(4, 4))
# x_ref [B, 8, 16, T'] → [B, num_ref_tokens, model_dim]

# In each transformer block:
class DO1TransformerBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        self.self_attn = nn.MultiheadAttention(model_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(model_dim, num_heads)
        self.ffn = FeedForward(model_dim)
        self.norm1 = AdaLN(model_dim)  # adaptive layer norm conditioned on timestep t
        self.norm2 = AdaLN(model_dim)
        self.norm3 = AdaLN(model_dim)

    def forward(self, x, ref_tokens, t_emb):
        # Self-attention over input tokens
        x = x + self.self_attn(self.norm1(x, t_emb), self.norm1(x, t_emb), self.norm1(x, t_emb))
        # Cross-attention to x_ref tokens
        x = x + self.cross_attn(self.norm2(x, t_emb), ref_tokens, ref_tokens)
        # Feed-forward
        x = x + self.ffn(self.norm3(x, t_emb))
        return x
```

### 1.4 Timestep Conditioning

Standard sinusoidal embedding → MLP → AdaLN modulation on each transformer block. Same as standard DiT.

### 1.5 Output

Unpatchify back to [B, 8, 16, T]. Output is the predicted velocity v_pred.

### 1.6 Model Size Target

~3.3B parameters. Scale model_dim and num_layers to hit this. Rough target:
- model_dim: 2048-3072
- num_heads: 32-48
- num_layers: 28-36
- FFN hidden: 4x model_dim

Verify parameter count and adjust.

### 1.7 CFG Dropout

During training, with probability 0.3, replace x_ref with zeros before patch embedding. This enables classifier-free guidance at inference.

```python
if training and random.random() < 0.3:
    x_ref = torch.zeros_like(x_ref)
```

---

## PART 2: FLOW MATCHING TRAINING

### 2.1 Flow Matching (Conditional Flow Matching / Rectified Flow)

```python
# Sample timestep uniformly
t = torch.rand(batch_size, device=device)  # [B] in [0, 1]

# Interpolate between noise and target
noise = torch.randn_like(z_target)
x_noisy = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * z_target

# Target velocity
v_target = z_target - noise

# Model prediction
v_pred = model(x_noisy, x_cond, x_ref, mask, t)

# Loss: MSE on velocity
loss = F.mse_loss(v_pred, v_target)
```

### 2.2 Inference (ODE solve)

At inference, use an ODE solver (Euler, midpoint, or DPM++) to integrate from t=0 (noise) to t=1 (data):

```python
# Euler method, N steps
x = torch.randn_like(z_target)  # start from noise
dt = 1.0 / N
for i in range(N):
    t = torch.full((B,), i * dt, device=device)
    v = model(x, x_cond, x_ref, mask, t)
    x = x + v * dt
# x is now the generated z_output
```

### 2.3 CFG at Inference

```python
v_cond = model(x, x_cond, x_ref, mask, t)
v_uncond = model(x, x_cond, torch.zeros_like(x_ref), mask, t)
v = v_uncond + cfg_scale * (v_cond - v_uncond)
```

---

## PART 3: DATALOADER

This is the most complex part. The dataloader must handle 7 training tasks with different data construction for each.

### 3.1 Dataset Structure

Assume the following directory structure:
```
/data/latents/
  session_001/
    drums.latent.npy        # [8, 16, T] float32
    drums.mean.npy          # [8, 16] float32 (temporal mean)
    drums.midi              # basic_pitch MIDI (for latent synth)
    bass.latent.npy
    bass.mean.npy
    bass.midi
    guitar_take1.latent.npy
    guitar_take2.latent.npy
    ...
  session_002/
    ...

/data/fx_pairs/
  pair_001/
    dry.latent.npy
    wet.latent.npy
  ...

/data/vst_synths/
  patch_001_midi_001.latent.npy
  patch_001_midi_002.latent.npy
  patch_002_midi_001.latent.npy
  ...

/data/instrument_labels.json   # { "session_001/drums.latent.npy": "drums", ... } for 74K files
```

### 3.2 Task Distribution

Each training step randomly selects a task:

```python
TASK_DISTRIBUTION = {
    'reconstruction': 0.35,   # corrupt(z_target), x_ref=same_session, mask=ones/partial
    'separation':     0.20,   # z_mix, x_ref=instrument_ref, mask=zeros
    'cross_instrument': 0.15, # z_synth or z_diff_inst, x_ref=same_inst_ref, mask=ones
    'fx':             0.10,   # z_wet, x_ref=dry_ref, mask=ones
    'generation':     0.10,   # zeros, x_ref=instrument_ref, mask=zeros
    'inpainting':     0.05,   # z_target with temporal gaps, mask=partial
    'synth_diversity': 0.05,  # z_VST_patch_A, x_ref=z_VST_patch_B, mask=ones
}
```

### 3.3 Reconstruction Task (35%)

```python
def get_reconstruction_sample(dataset):
    # Pick random target stem
    z_target = load_random_stem()
    session_stems = get_stems_in_same_session(z_target)

    # Pick x_ref from same session (any other stem in the folder)
    x_ref = load_random_from(session_stems, exclude=z_target)
    # If session has only one stem, use z_target itself or random from dataset
    if x_ref is None:
        x_ref = load_random_stem()

    # Apply random corruption to z_target to create x_cond
    x_cond, mask = apply_corruption(z_target)

    return x_cond, x_ref, z_target, mask
```

### 3.4 Corruption Pipeline (6 types, applied within reconstruction task)

```python
CORRUPTION_DISTRIBUTION = {
    'light_noise':       0.20,
    'channel_dropout':   0.20,
    'temporal_masking':  0.15,
    'mean_swap':         0.15,
    'full_substitution': 0.15,
    'blended':           0.15,
}

def apply_corruption(z_target):
    corruption_type = random_choice(CORRUPTION_DISTRIBUTION)

    if corruption_type == 'light_noise':
        sigma = random.uniform(0.05, 0.2)
        x_cond = z_target + sigma * torch.randn_like(z_target)
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])  # [1, 16, T]

    elif corruption_type == 'channel_dropout':
        # Zero out 30-50% of the 8 latent channels randomly
        num_drop = random.randint(3, 4)  # out of 8 channels
        drop_channels = random.sample(range(8), num_drop)
        x_cond = z_target.clone()
        x_cond[drop_channels] = 0.0
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    elif corruption_type == 'temporal_masking':
        # Zero out random temporal spans (5-20 frames each, 2-5 spans)
        x_cond = z_target.clone()
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])
        T = z_target.shape[2]
        num_spans = random.randint(2, 5)
        for _ in range(num_spans):
            span_len = random.randint(5, 20)
            start = random.randint(0, max(0, T - span_len))
            x_cond[:, :, start:start+span_len] = 0.0
            mask[:, :, start:start+span_len] = 0.0

    elif corruption_type == 'mean_swap':
        # Replace temporal mean with a random stem's mean
        z_random = load_random_stem()
        mean_target = z_target.mean(dim=-1, keepdim=True)   # [8, 16, 1]
        mean_random = z_random.mean(dim=-1, keepdim=True)
        x_cond = z_target - mean_target + mean_random
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    elif corruption_type == 'full_substitution':
        # Replace x_cond with a completely different recording
        x_cond = load_random_stem()
        # Time-match: truncate or pad to match z_target length
        x_cond = match_length(x_cond, z_target.shape[2])
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    elif corruption_type == 'blended':
        alpha = random.uniform(0.3, 0.7)
        z_random = load_random_stem()
        z_random = match_length(z_random, z_target.shape[2])
        x_cond = alpha * z_target + (1 - alpha) * z_random
        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

    return x_cond, mask
```

### 3.5 Separation Task (20%)

Mixes are constructed ON THE FLY by summing stems from the same song. Never sum alternate takes of the same part.

```python
def get_separation_sample(dataset):
    # Pick target stem
    z_target = load_random_stem()
    song_stems = get_other_stems_in_same_song(z_target)

    # Filter out alternate takes of the same part
    # Rule: if high timbre sim (mean cosine > 0.85) AND high content sim
    # (cross-correlation on channel 6 > 0.7), it's an alternate take — skip
    valid_stems = filter_alternate_takes(song_stems, z_target)

    if len(valid_stems) == 0:
        # Fallback: use stems from a different random song
        valid_stems = get_random_stems_from_different_song(n=random.randint(2, 4))

    # Determine mix difficulty
    difficulty = random_choice({'easy': 0.60, 'hard': 0.25, 'dense': 0.15})

    if difficulty == 'easy':
        # 2-4 stems with distinct instruments (use instrument labels if available)
        mix_stems = select_distinct_instruments(valid_stems, n=random.randint(2, 4))
    elif difficulty == 'hard':
        # 2-4 stems from same instrument family
        mix_stems = select_same_family(valid_stems, n=random.randint(2, 4))
    else:  # dense
        # 5-10 stems including similar instruments
        mix_stems = select_up_to(valid_stems, n=random.randint(5, min(10, len(valid_stems))))

    # Construct mix by summing in latent space
    z_mix = z_target.clone()
    for stem in mix_stems:
        z_stem = load_stem(stem)
        z_stem = match_length(z_stem, z_target.shape[2])
        z_mix = z_mix + z_stem

    # x_ref: a clip of the target instrument from elsewhere in dataset
    # Use instrument label if available, or sample from same session
    x_ref = find_same_instrument_ref(z_target)

    mask = torch.zeros(1, z_target.shape[1], z_target.shape[2])  # all zeros for separation
    return z_mix, x_ref, z_target, mask
```

### 3.6 Cross-Instrument Task (15%)

```python
def get_cross_instrument_sample(dataset, latent_synth):
    z_target = load_random_stem()
    session_stems = get_stems_in_same_session(z_target)

    option = random.choice(['latent_synth', 'natural_cross_instrument'])

    if option == 'latent_synth':
        # Load cached basic_pitch MIDI for this stem
        midi = load_midi_for_stem(z_target)
        # Render through latent synth with RANDOM params (on the fly, ~2.8ms)
        random_vcf = sample_random_vcf_params()
        random_vca = sample_random_vca_params()
        z_synth = latent_synth.render(midi, vcf=random_vcf, vca=random_vca)
        x_cond = match_length(z_synth, z_target.shape[2])

        # x_ref from same session (same instrument as target)
        x_ref = load_random_from(session_stems, exclude=z_target)
        if x_ref is None:
            x_ref = z_target  # self-reference fallback

    elif option == 'natural_cross_instrument':
        # Sample any other stem from same session folder
        # (might be same instrument different part, or different instrument)
        other_stem = load_random_from(session_stems, exclude=z_target)
        if other_stem is None:
            # Fallback to latent synth
            return get_cross_instrument_sample_latent_synth(dataset, latent_synth, z_target)
        x_cond = match_length(other_stem, z_target.shape[2])
        x_ref = load_random_from(session_stems, exclude=[z_target, other_stem])
        if x_ref is None:
            x_ref = z_target

    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])  # ones for transfer
    return x_cond, x_ref, z_target, mask
```

### 3.7 FX Task (10%)

```python
def get_fx_sample(dataset, fx_processors):
    # Select source tier
    source = random_choice({'real': 0.40, 'augmented_real': 0.30, 'synthetic': 0.30})
    # Track weight for loss scaling
    if source == 'real':
        loss_weight = 1.0
    elif source == 'augmented_real':
        loss_weight = 0.5
    else:
        loss_weight = 0.3

    if source == 'real':
        # Load real dry/wet pair
        pair = load_random_fx_pair()
        z_dry, z_wet = pair['dry'], pair['wet']
    elif source == 'augmented_real':
        # Load real dry, apply synthetic FX
        pair = load_random_fx_pair()
        z_dry = pair['dry']
        z_wet = apply_random_fx_chain(z_dry, fx_processors)
    else:
        # Fully synthetic: apply FX to random stem
        z_dry = load_random_stem()
        z_wet = apply_random_fx_chain(z_dry, fx_processors)

    # Temporal FX application (50% of the time)
    if random.random() < 0.5:
        z_wet = apply_temporal_fx_mask(z_dry, z_wet)

    # Random direction: 50% removal (wet→dry), 50% application (dry→wet)
    if random.random() < 0.5:
        # FX removal
        x_cond = z_wet
        target = z_dry
        x_ref = load_random_dry_reference()  # some dry recording for "dry character"
    else:
        # FX application / matching
        x_cond = z_dry
        target = z_wet
        x_ref = load_random_wet_reference()  # some wet recording for "wet character"

    mask = torch.ones(1, target.shape[1], target.shape[2])
    return x_cond, x_ref, target, mask, loss_weight


def apply_random_fx_chain(z_dry, fx_processors):
    """Apply 1-4 random FX in sequence. Requires DCAE decode/encode."""
    # Available FX: eq, compressor, distortion, reverb, chorus, delay, pitch_shift
    num_fx = random.randint(1, 4)
    fx_chain = random.sample(list(fx_processors.keys()), num_fx)

    audio = dcae_decode(z_dry)
    for fx_name in fx_chain:
        params = sample_random_params(fx_name)
        audio = fx_processors[fx_name](audio, params)
    z_wet = dcae_encode(audio)
    return z_wet


def apply_temporal_fx_mask(z_dry, z_wet):
    """Apply FX only to random temporal regions."""
    T = z_dry.shape[2]
    fx_mask = torch.zeros(1, 1, T)

    # Generate 2-5 random wet regions covering 30-70% of frames
    num_regions = random.randint(2, 5)
    target_coverage = random.uniform(0.3, 0.7)
    frames_to_fill = int(T * target_coverage)
    filled = 0
    for _ in range(num_regions):
        span = min(random.randint(5, T // 3), frames_to_fill - filled)
        start = random.randint(0, max(0, T - span))
        fx_mask[:, :, start:start+span] = 1.0
        filled += span
        if filled >= frames_to_fill:
            break

    # Blend dry and wet based on mask
    z_partial = z_dry * (1 - fx_mask) + z_wet * fx_mask
    return z_partial
```

### 3.8 Generation Task (10%)

```python
def get_generation_sample(dataset):
    z_target = load_random_stem()

    # x_cond = zeros (no content provided)
    x_cond = torch.zeros_like(z_target)

    # x_ref = some clip of the same instrument from anywhere
    x_ref = find_same_instrument_ref(z_target)

    mask = torch.zeros(1, z_target.shape[1], z_target.shape[2])  # zeros for generation
    return x_cond, x_ref, z_target, mask
```

### 3.9 Inpainting Task (5%)

```python
def get_inpainting_sample(dataset):
    z_target = load_random_stem()
    session_stems = get_stems_in_same_session(z_target)

    x_cond = z_target.clone()
    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])
    T = z_target.shape[2]

    # Zero out 1-3 contiguous regions (10-30% of total length each)
    num_gaps = random.randint(1, 3)
    for _ in range(num_gaps):
        gap_len = random.randint(int(T * 0.1), int(T * 0.3))
        start = random.randint(0, max(0, T - gap_len))
        x_cond[:, :, start:start+gap_len] = 0.0
        mask[:, :, start:start+gap_len] = 0.0

    x_ref = load_random_from(session_stems, exclude=z_target)
    if x_ref is None:
        x_ref = z_target  # self-reference for style

    return x_cond, x_ref, z_target, mask
```

### 3.10 Synth Diversity Task (5%)

```python
def get_synth_diversity_sample(vst_dataset):
    """
    VST dataset has structure: {midi_id: {patch_id: z_latent}}
    Same MIDI rendered through different patches = cross-timbre pair.
    """
    # Pick a random MIDI and two different patches
    midi_id = random.choice(list(vst_dataset.keys()))
    patches = list(vst_dataset[midi_id].keys())
    if len(patches) < 2:
        # Fallback: pick different MIDI for x_ref
        patch_a = patches[0]
        midi_id_b = random.choice([m for m in vst_dataset.keys() if m != midi_id])
        patch_b = random.choice(list(vst_dataset[midi_id_b].keys()))
        z_target = vst_dataset[midi_id][patch_a]
        x_ref_melody = vst_dataset[midi_id_b][patch_b] if patch_b == patch_a else vst_dataset[midi_id_b][patch_b]
    else:
        patch_a, patch_b = random.sample(patches, 2)
        z_target = vst_dataset[midi_id][patch_a]

    # x_cond: target patch playing a DIFFERENT melody (or different patch, same melody)
    # x_ref: target patch playing a different melody (provides timbre)
    midi_id_ref = random.choice([m for m in vst_dataset.keys() if m != midi_id])
    if patch_a in vst_dataset[midi_id_ref]:
        x_ref = vst_dataset[midi_id_ref][patch_a]  # same patch, different melody
    else:
        x_ref = vst_dataset[midi_id_ref][random.choice(list(vst_dataset[midi_id_ref].keys()))]

    # x_cond: different patch, same melody (cross-timbre)
    x_cond = vst_dataset[midi_id][patch_b]  # same melody, wrong patch

    mask = torch.ones(1, z_target.shape[1], z_target.shape[2])
    return x_cond, x_ref, z_target, mask
```

---

## PART 4: TRAINING LOOP

```python
def train_step(model, optimizer, batch):
    """
    batch contains: x_cond, x_ref, z_target, mask, loss_weight
    All tensors already on device.
    """
    B = z_target.shape[0]

    # CFG dropout: 30% chance replace x_ref with zeros
    cfg_mask = (torch.rand(B, 1, 1, 1, device=device) > 0.3).float()
    x_ref = x_ref * cfg_mask

    # Sample timestep
    t = torch.rand(B, device=device)

    # Flow matching interpolation
    noise = torch.randn_like(z_target)
    x_noisy = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * z_target
    v_target = z_target - noise

    # Forward pass
    v_pred = model(x_noisy, x_cond, x_ref, mask, t)

    # Weighted MSE loss
    loss = (loss_weight[:, None, None, None] * (v_pred - v_target) ** 2).mean()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


def train(model, dataset, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, config.total_steps)

    for step in range(config.total_steps):
        # Sample task
        task = random_choice(TASK_DISTRIBUTION)

        # Get batch for this task
        if task == 'reconstruction':
            batch = collate([get_reconstruction_sample(dataset) for _ in range(config.batch_size)])
        elif task == 'separation':
            batch = collate([get_separation_sample(dataset) for _ in range(config.batch_size)])
        elif task == 'cross_instrument':
            batch = collate([get_cross_instrument_sample(dataset, latent_synth) for _ in range(config.batch_size)])
        elif task == 'fx':
            batch = collate([get_fx_sample(dataset, fx_processors) for _ in range(config.batch_size)])
        elif task == 'generation':
            batch = collate([get_generation_sample(dataset) for _ in range(config.batch_size)])
        elif task == 'inpainting':
            batch = collate([get_inpainting_sample(dataset) for _ in range(config.batch_size)])
        elif task == 'synth_diversity':
            batch = collate([get_synth_diversity_sample(vst_dataset) for _ in range(config.batch_size)])

        loss = train_step(model, optimizer, batch)
        scheduler.step()

        # Logging
        if step % config.log_every == 0:
            log(step=step, task=task, loss=loss, lr=scheduler.get_last_lr()[0])

        # Checkpointing
        if step % config.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step)

        # Evaluation
        if step % config.eval_every == 0:
            run_evaluation(model, eval_dataset)
```

---

## PART 5: COLLATION / BATCHING

Variable-length latents require padding within a batch.

```python
def collate(samples):
    """
    Each sample is (x_cond, x_ref, z_target, mask) or
    (x_cond, x_ref, z_target, mask, loss_weight)
    
    Tensors have shape [8, 16, T] where T varies.
    Pad all to max T in the batch.
    """
    max_T_target = max(s[2].shape[2] for s in samples)
    max_T_ref = max(s[1].shape[2] for s in samples)

    x_cond_batch = []
    x_ref_batch = []
    z_target_batch = []
    mask_batch = []
    weight_batch = []

    for sample in samples:
        x_cond, x_ref, z_target, mask = sample[:4]
        loss_weight = sample[4] if len(sample) > 4 else 1.0

        # Pad target-length tensors
        x_cond_batch.append(pad_to(x_cond, max_T_target))
        z_target_batch.append(pad_to(z_target, max_T_target))
        mask_batch.append(pad_to(mask, max_T_target))  # pad mask with 0 (don't generate in padding)

        # Pad ref-length tensor
        x_ref_batch.append(pad_to(x_ref, max_T_ref))

        weight_batch.append(loss_weight)

    return {
        'x_cond': torch.stack(x_cond_batch),
        'x_ref': torch.stack(x_ref_batch),
        'z_target': torch.stack(z_target_batch),
        'mask': torch.stack(mask_batch),
        'loss_weight': torch.tensor(weight_batch),
    }


def pad_to(tensor, target_T):
    """Zero-pad along last dim (time) to target_T."""
    T = tensor.shape[-1]
    if T >= target_T:
        return tensor[..., :target_T]
    pad_size = target_T - T
    return F.pad(tensor, (0, pad_size), value=0.0)
```

---

## PART 6: UTILITY FUNCTIONS TO IMPLEMENT

```python
def load_random_stem() -> Tensor:
    """Load a random .latent.npy file from the dataset. Returns [8, 16, T]."""
    pass

def get_stems_in_same_session(z_target) -> List[str]:
    """Return paths to all stems in the same session folder as z_target."""
    pass

def load_random_from(stems, exclude=None) -> Optional[Tensor]:
    """Load a random stem from the list, excluding specified stems. Returns None if empty."""
    pass

def match_length(z, target_T) -> Tensor:
    """Truncate or zero-pad z along time dimension to target_T frames."""
    pass

def find_same_instrument_ref(z_target) -> Tensor:
    """Find a stem of the same instrument from elsewhere in the dataset.
    Use instrument labels if available (74K labeled files).
    Otherwise use z_mean cosine similarity or random same-session stem."""
    pass

def filter_alternate_takes(stems, z_target) -> List[str]:
    """Remove stems that are alternate takes of z_target's part.
    Filter: high timbre sim (mean cosine > 0.85) AND high content sim 
    (cross-correlation on channel 6 > 0.7) = alternate take, remove.
    Different parts from same instrument are valid, keep them."""
    pass

def select_distinct_instruments(stems, n) -> List[str]:
    """Select n stems with distinct instrument labels/timbres for easy mixes."""
    pass

def select_same_family(stems, n) -> List[str]:
    """Select n stems from the same instrument family for hard mixes."""
    pass

def load_random_fx_pair() -> dict:
    """Load a random real dry/wet pair. Returns {'dry': Tensor, 'wet': Tensor}."""
    pass

def load_random_dry_reference() -> Tensor:
    """Load a random dry recording as FX reference."""
    pass

def load_random_wet_reference() -> Tensor:
    """Load a random wet recording as FX reference."""
    pass

def load_midi_for_stem(z_target) -> MidiData:
    """Load the cached basic_pitch MIDI file for this stem."""
    pass

def sample_random_vcf_params() -> dict:
    """Random VCF parameters for latent synth."""
    pass

def sample_random_vca_params() -> dict:
    """Random VCA parameters for latent synth."""
    pass
```

---

## PART 7: TRAINING CONFIG

```python
@dataclass
class TrainingConfig:
    # Model
    model_dim: int = 2560
    num_heads: int = 40
    num_layers: int = 32
    patch_size: tuple = (4, 4)
    
    # Training
    batch_size: int = 8           # per GPU, adjust for VRAM
    gradient_accumulation: int = 4 # effective batch = 32
    lr: float = 1e-4
    warmup_steps: int = 5000
    total_steps: int = 500_000    # adjust based on convergence
    max_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data
    max_T: int = 512              # max temporal frames (~47s at 10.8fps)
    
    # Logging
    log_every: int = 100
    save_every: int = 5000
    eval_every: int = 10000
    
    # Paths
    data_dir: str = "/data/latents"
    fx_dir: str = "/data/fx_pairs"
    vst_dir: str = "/data/vst_synths"
    checkpoint_dir: str = "/checkpoints/do1"
```

---

## PART 8: EVALUATION

```python
def run_evaluation(model, eval_dataset):
    """Run per-task evaluation metrics."""
    model.eval()
    with torch.no_grad():
        # 1. Reconstruction: MSE in latent space
        recon_mse = eval_reconstruction(model, eval_dataset)
        
        # 2. Separation: decode stems, compute SDR against ground truth
        separation_sdr = eval_separation(model, eval_dataset)
        
        # 3. Timbre transfer: cosine sim of output mean vs x_ref mean (timbre match)
        #    + chroma similarity of output vs x_cond (pitch preservation)
        timbre_sim, pitch_sim = eval_timbre_transfer(model, eval_dataset)
        
        # 4. FX removal: MSE against dry ground truth
        fx_mse = eval_fx_removal(model, eval_dataset)
        
        # 5. Generation: FAD (Fréchet Audio Distance) against real distribution
        gen_fad = eval_generation_fad(model, eval_dataset)
        
        log_metrics({
            'recon_mse': recon_mse,
            'separation_sdr': separation_sdr,
            'timbre_sim': timbre_sim,
            'pitch_sim': pitch_sim,
            'fx_mse': fx_mse,
            'gen_fad': gen_fad,
        })
    model.train()
```

---

## CRITICAL IMPLEMENTATION NOTES

1. **Latent shape is [8, 16, T]** — 8 channels, 16 frequency bins, T time frames at 10.8fps. NOT [128, T]. The 8×16=128 is the full latent dim but organized as a 2D spatial grid for the DiT.

2. **PatchEmbed operates on [16, T]** as a 2D spatial grid — 16 is frequency, T is time. With patch_size (4,4): 4 freq patches × T/4 time patches.

3. **x_ref can have different temporal length T' from x_cond/z_target T.** The cross-attention handles this naturally (queries from input tokens, keys/values from ref tokens).

4. **Flow matching, NOT DDPM.** Use conditional flow matching (rectified flow) — linear interpolation between noise and data, velocity prediction, ODE sampling. NOT epsilon prediction, NOT noise scheduling.

5. **FX chain application requires DCAE decode→FX→encode.** This is expensive (~100ms+). For synthetic FX pairs, consider doing this in a background data worker or precomputing a cache of FX'd versions that gets refreshed periodically.

6. **Latent synth renders on GPU in 2.8ms per full MIDI.** Always do this on the fly with random params. Never precompute.

7. **Loss weighting:** FX task has per-sample weights (1.0 for real, 0.5 for augmented, 0.3 for synthetic). All other tasks use weight 1.0.

8. **AdaLN (Adaptive Layer Normalization):** Timestep embedding modulates scale and shift of layer norm in each transformer block. Standard DiT approach. Look at the DiT paper or ACE-Step implementation.

9. **Mixed precision training (bf16/fp16):** Essential for 3.3B params. Use PyTorch AMP or DeepSpeed.

10. **Distributed training:** Use FSDP or DeepSpeed ZeRO-3 for multi-GPU. The model won't fit on a single GPU in fp32.
