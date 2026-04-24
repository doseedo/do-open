"""
End-to-end smoke test:
  - load frozen Oobleck VAE from ACE-Step handler
  - build LatentSpliceDataset on /mnt/data2/Latents2/protools
  - pull one batch
  - run one training step
  - run one inference call
Prints shapes + losses; nonzero exit on failure.
"""
import sys, traceback, torch
sys.path.insert(0, "/home/arlo/do2")
sys.path.insert(0, "/scratch/ACE-Step-1.5")

from acestep.handler import AceStepHandler
from latent_editor.dataset import LatentSpliceDataset, collate
from latent_editor.model import LatentEditor
from latent_editor.train import latent_loss, multi_res_stft_loss
from latent_editor.dataset import SAMPLES_PER_FRAME

def main():
    print("init handler...")
    h = AceStepHandler()
    h.initialize_service(
        project_root="/scratch/ACE-Step-1.5",
        config_path="acestep-v15-sft",
        device="cuda",
    )
    vae = h.vae
    for p in vae.parameters():
        p.requires_grad = False
    vae.eval()

    print("build dataset...")
    ds = LatentSpliceDataset(
        roots=["/mnt/data2/Latents2/protools"],
        vae=vae, win_frames=64, device="cuda", seed=0,
    )
    print(f"  files indexed: {len(ds.files)}")

    print("fetch one item...")
    item = ds[0]
    for k, v in item.items():
        print(f"  {k}: {tuple(v.shape) if hasattr(v,'shape') else v}")
    batch = collate([ds[0], ds[1]])
    print(f"  batch L_naive: {tuple(batch.L_naive.shape)}  L_target: {tuple(batch.L_target.shape)}")

    print("build model + one training step...")
    model = LatentEditor(max_len=64).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    L_naive  = batch.L_naive.cuda()
    L_target = batch.L_target.cuda()
    mask     = batch.mask.cuda()
    phase    = batch.phase.cuda()
    wav_tgt  = batch.wav_target.cuda()
    cf       = int(batch.cut_frame[0].item())

    pred = model(L_naive, mask, phase)
    l_lat = latent_loss(pred, L_target, mask)

    r = 8
    T = pred.shape[1]
    lo, hi = max(0, cf - r), min(T, cf + r + 1)
    sub_pred = pred[:, lo:hi].transpose(1, 2).to(torch.bfloat16)
    wav_pred = vae.decode(sub_pred).sample.float()
    s_lo, s_hi = lo * SAMPLES_PER_FRAME, hi * SAMPLES_PER_FRAME
    n = min(wav_pred.shape[-1], wav_tgt.shape[-1] - s_lo, s_hi - s_lo)
    l_stft = multi_res_stft_loss(wav_pred[..., :n], wav_tgt[:, :, s_lo:s_lo + n])
    loss = l_lat + l_stft
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}  lat={l_lat.item():.4f}  stft={l_stft.item():.4f}")

    print("inference round-trip...")
    from latent_editor.infer import LatentEditorRuntime
    # save+load to exercise ckpt path
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "ckpt.pt")
        torch.save({"model": model.state_dict(), "args": {"win_frames": 64}}, cp)
        rt = LatentEditorRuntime(cp)
        La = batch.L_naive[0].cpu()
        Lb = batch.L_target[0].cpu()
        out = rt.edit(La, Lb, cut_sample=cf * SAMPLES_PER_FRAME + 731)
    print(f"  edit out shape: {tuple(out.shape)}")
    print("OK")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
