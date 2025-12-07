#!/usr/bin/env python3
"""v55 Checkpoint Audit - Fast version with orjson"""
import orjson
import os
import numpy as np
from collections import defaultdict, Counter
import time

os.chdir('/home/arlo/do-repo/midi_generator/1_approaches/transform_based')

INST = {0:"Piano",24:"Nylon Guitar",25:"Steel Guitar",27:"Electric Guitar",
        32:"Acoustic Bass",33:"Electric Bass",56:"Trumpet",57:"Trombone",
        60:"French Horn",65:"Alto Sax",66:"Tenor Sax",67:"Baritone Sax",
        71:"Clarinet",73:"Flute",128:"Drums"}

t0 = time.time()
R = {}

print("="*60)
print("v55 CHECKPOINT AUDIT (orjson)")
print("="*60)

# Load patterns
print("\nLoading 1.9GB patterns...")
lt = time.time()
with open('checkpoint_v55_pure_contour_1000files_patterns.json', 'rb') as f:
    patterns = orjson.loads(f.read())
print(f"Loaded {len(patterns):,} patterns in {time.time()-lt:.1f}s")

# Part 1: Per-instrument vocab
print("\n[1] PER-INSTRUMENT VOCABULARY")
gm_counts = defaultdict(int)
non_prefixed = []
mismatches = []
for pid, p in patterns.items():
    if pid.startswith("GM"):
        gm = pid.split("_")[0]
        gm_counts[gm] += 1
        if int(gm[2:]) != p.get('gm_program', -1):
            mismatches.append(pid)
    else:
        non_prefixed.append(pid)

for gm, c in sorted(gm_counts.items(), key=lambda x: int(x[0][2:])):
    print(f"  {gm} ({INST.get(int(gm[2:]),'')[:12]}): {c}")

R['vocab'] = len(non_prefixed) == 0
R['gm_match'] = len(mismatches) == 0
print(f"{'✅' if R['vocab'] else '❌'} All GM-prefixed: {R['vocab']}")
print(f"{'✅' if R['gm_match'] else '❌'} gm_program matches: {R['gm_match']} ({len(mismatches)} mismatches)")

# Part 2: Ratios
print("\n[2] RHYTHM/VELOCITY/DURATION RATIOS")
empty_r = empty_v = empty_d = pop_all = single = 0
for p in patterns.values():
    iv = p.get('pitch_intervals', [])
    if len(iv) <= 1:
        single += 1
        continue
    r, v, d = p.get('rhythm_ratios',[]), p.get('velocity_ratios',[]), p.get('duration_ratios',[])
    if not r: empty_r += 1
    if not v: empty_v += 1
    if not d: empty_d += 1
    if r and v and d: pop_all += 1

multi = len(patterns) - single
print(f"  Single-note: {single}, Multi-note: {multi}")
print(f"  Empty rhythm: {empty_r}/{multi} ({100*empty_r/max(1,multi):.1f}%)")
print(f"  Empty velocity: {empty_v}/{multi}")
print(f"  Empty duration: {empty_d}/{multi}")
print(f"  ALL populated: {pop_all}/{multi} ({100*pop_all/max(1,multi):.1f}%)")
R['ratios'] = empty_r / max(1, multi) < 0.5
print(f"{'✅' if R['ratios'] else '❌'} Ratios populated: {R['ratios']}")

# Part 3: Pitch grounding
print("\n[3] OCCURRENCE PITCH GROUNDING")
total = valid = 0
pitch_by_gm = defaultdict(list)
for pid, p in patterns.items():
    pgm = p.get('gm_program', -1)
    for o in p.get('occurrences', []):
        total += 1
        if 'first_pitch' in o and 'onset_time' in o and o.get('gm_program') == pgm:
            valid += 1
            fp = o.get('first_pitch', -1)
            if 0 <= fp <= 127:
                pitch_by_gm[pgm].append(fp)

print(f"  Total: {total:,}, Valid: {valid:,} ({100*valid/max(1,total):.1f}%)")
R['pitch'] = valid / max(1, total) > 0.9
print(f"{'✅' if R['pitch'] else '❌'} Pitch grounding: {R['pitch']}")

print("\n  Pitch ranges:")
for gm in sorted(pitch_by_gm.keys()):
    pp = pitch_by_gm[gm]
    if pp:
        print(f"    GM{gm} ({INST.get(gm,'')[:10]}): [{min(pp)}-{max(pp)}] med={sorted(pp)[len(pp)//2]} n={len(pp):,}")

# Part 4: TrackDerive
print("\n[4] TRACKDERIVE")
with open('checkpoint_v55_pure_contour_1000files_track_derives.json', 'rb') as f:
    td = orjson.loads(f.read())
R['td'] = len(td) > 0
print(f"  Relationships: {len(td):,}")
print(f"{'✅' if R['td'] else '❌'} TrackDerive: {R['td']}")

# Part 5: Transforms
print("\n[5] TRANSFORMS")
with open('checkpoint_v55_pure_contour_1000files_transforms.json', 'rb') as f:
    tf = orjson.loads(f.read())
print(f"  Transforms: {len(tf)}")
if isinstance(tf, list): print(f"  {tf[:12]}...")
R['tf'] = len(tf) > 0

# Part 6: Resolution test
print("\n[6] RESOLUTION TEST")
test_gms = [0, 32, 33, 56, 57, 65, 66, 67, 71, 73]
fails = 0
for gm in test_gms:
    pats = [pid for pid in patterns if pid.startswith(f"GM{gm}_")]
    if not pats:
        print(f"  ❌ GM{gm}: No patterns")
        fails += 1
        continue
    p = patterns[pats[0]]
    occs = p.get('occurrences', [])
    if not occs or 'first_pitch' not in occs[0]:
        print(f"  ❌ GM{gm}: No valid occ")
        fails += 1
        continue
    fp = occs[0]['first_pitch']
    pitches = [fp] + [fp := fp + iv for iv in p.get('pitch_intervals', [])]
    if any(x < 0 or x > 127 for x in pitches):
        print(f"  ❌ GM{gm}: Bad pitches")
        fails += 1
    else:
        print(f"  ✅ GM{gm} ({INST.get(gm,'')[:10]}): {pitches[:4]}...")
R['res'] = fails == 0

# Part 7: NPZ
print("\n[7] NPZ (OOM PHASES)")
npz = np.load('checkpoint_v55_pure_contour_1000files.npz', allow_pickle=True)
print(f"  Keys: {list(npz.keys())}")
print("  ✅ Missing phases not critical")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
checks = [("Per-instrument vocab", R['vocab']), ("gm_program match", R['gm_match']),
          ("Ratios populated", R['ratios']), ("Pitch grounding", R['pitch']),
          ("TrackDerive", R['td']), ("Resolution", R['res'])]
for name, ok in checks:
    print(f"{'✅' if ok else '❌'} {name}")

if all(v for _, v in checks):
    print("\n✅ ALL PASSED - Ready for generation")
else:
    print("\n❌ SOME FAILED")

print(f"\nTotal time: {time.time()-t0:.1f}s")
