"""End-to-end smoke for the latent → BasicPitch student."""
import sys, traceback, torch
sys.path.insert(0, "/home/arlo/do2")

from latent_pitch.dataset import LatentMidiPairDataset, collate_pitch
from latent_pitch.model import LatentBasicPitchStudent
from latent_pitch.train import masked_bce, masked_l1_active

def main():
    print("dataset...")
    ds = LatentMidiPairDataset(win_frames=256, seed=0)
    print(f"  dates: {len(ds.dates)}")
    batch = collate_pitch([ds[0], ds[1]])
    print(f"  latent {tuple(batch['latent'].shape)} onset {tuple(batch['onset'].shape)}")

    print("model + one training step...")
    m = LatentBasicPitchStudent(max_len=256).to("cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    L     = batch["latent"].cuda()
    onset = batch["onset"].cuda()
    frame = batch["frame"].cuda()
    vel   = batch["velocity"].cuda()
    mask  = batch["mask"].cuda()
    pw_o = torch.tensor([20.0], device="cuda")
    pw_f = torch.tensor([5.0],  device="cuda")
    out = m(L)
    l_on  = masked_bce(out["onset_logits"], onset, mask, pw_o)
    l_fr  = masked_bce(out["frame_logits"], frame, mask, pw_f)
    l_vel = masked_l1_active(out["velocity"], vel, mask, frame)
    loss = l_on + l_fr + 0.5 * l_vel
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  loss={loss.item():.4f}  on={l_on.item():.4f}  fr={l_fr.item():.4f}  vel={l_vel.item():.4f}")

    print("inference round-trip...")
    import tempfile, os
    from latent_pitch.infer import LatentPitchRuntime
    with tempfile.TemporaryDirectory() as d:
        cp = os.path.join(d, "ckpt.pt")
        torch.save({"model": m.state_dict(), "args": {}}, cp)
        rt = LatentPitchRuntime(cp)
        pm = rt.transcribe(batch["latent"][0])
    print(f"  produced PrettyMIDI: {sum(len(i.notes) for i in pm.instruments)} notes")
    print("OK")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
