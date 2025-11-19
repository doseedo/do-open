import os
import json
import re
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- USER CONFIGURATION ---

# The text file containing a list of all audio file paths, one per line.
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths3.txt")

# The output JSON file where the grouped drum kits will be saved.
OUTPUT_JSON = Path("drum_groups.json")

# Minimum duration in seconds for a track to be considered for grouping.
MIN_DURATION_S = 5.0

# How close in seconds two tracks need to be to be considered the same duration.
DURATION_TOLERANCE = 0.01

# --- KEYWORD CONFIGURATION ---

# POSITIVE: Keywords to identify potential drum tracks.
DRUM_KEYWORDS_TEXT = """
    Kick KICK KickIn KickOut KickSub Kik Kck KIK kick kik
    Snare Snr SNR Sn SN SnrTop SnrBtm SNARE snare sn
    HiHat HH ClosedHat OpenHat Hat HAT hihat
    Tom RackTom FloorTom RTom FTom TOM
    Cymbal Cym Crash Ride Splash China Stack CYM CYMBAL
    OH Overhead OHL OHR OVERHEAD
    Perc Tamb Cowbell Clap Shaker Triangle PERC CLAP
    Drum Drums Drumkit Kit KIT DRUM DRUMKIT
"""
DRUM_KEYWORDS = DRUM_KEYWORDS_TEXT.strip().split()
drum_pattern = re.compile(r"|".join(re.escape(word) for word in DRUM_KEYWORDS), re.IGNORECASE)


# NEGATIVE: Keywords for other instruments to EXCLUDE a group.
NEGATIVE_KEYWORDS_TEXT = """
    # Bass and guitar
    Bass Basses Bassline Basslines ElectricBass ElectricBasses EBass EBasses AcousticBass AcousticBasses
    Guitar Guitars EGTR AGTR Acoustic Acoustics Electric Electrics BassGuitar BassGuitars BassGtr BassGtrs
    Gtr Gtrs Guit Guits Guitars BassAmp BassAmps GuitarAmp GuitarAmps Distortion Overdrive

    # Keys and keyboard instruments
    Piano Pianos GrandPiano GrandPianos UprightPiano UprightPianos
    Keys Keyboard Keyboards Rhodes Wurli Wurlitzer Wurlitzers
    Organ Organs Hammond Synth Synths Synthesizer Synthesizers
    SynthPad SynthPads Pad Pads Pluck Plucks Clav Clavs Clavinet Clavinets
    Harpsichord Harpsichords Celesta Celestas MalletKeys

    # Leads, melodies, arpeggios
    Lead Leads Arp Arps Arpeggio Arpeggios Bell Bells Chime Chimes
    Gliss Glisses Glissando Glissandos Glide Glides Portamento Portamentos
    Solo Solos Melody Melodies Melodic

    # Strings family
    String Strings StringSection StringSections
    Violin Violins Viola Violas Cello Cellos DoubleBass DoubleBasses
    Contrabass Contrabasses BassViolin BassViolins ViolinSolo ViolinSolos
    CelloSolo CelloSolos ViolaSolo ViolaSolos OrchestraStrings

    # Brass family
    Brass Brasses Trumpet Trumpets Trombone Trombones Tuba Tubas
    Horn Horns FrenchHorn FrenchHorns Cornet Cornets Flugelhorn Flugelhorns Euphonium Euphoniums

    # Woodwinds and others
    Sax Saxes Saxophone Saxophones Flute Flutes Clarinet Clarinets
    Oboe Oboes Bassoon Bassoons Piccolo Piccolos Recorder Recorders
    Whistle Whistles PanFlute PanFlutes

    # Vocals and choir
    Vocal Vocals Voice Voices Choir Choirs Choral Chorals
    BGV BGVs BackingVocals LeadVox LeadVoxes Harmony Harmonies
    Vox Voxes Harmonies Singing Shout Shouts Yell Yells Rap Raps SpokenWord

    # Percussion (non-drum) and effects
    FX SFX Effect Effects Ambience Ambiences Atmosphere Atmospheres
    Noise Noises Wind Winds Rain Crowd Crowds FieldRecording FieldRecordings
    RoomTone RoomTones Static Statics Sweep Sweeps Whoosh Whooshes
    Impact Impacts Hit Hits Scratch Scratches

    # Sub-bass, 808, bass synths (non-drums)
    808 808s Sub Subs SubBass SubBasses SubSynth SubSynths
    Tone Tones Note Notes Drone Drones

    # Tuned percussion and melodic percussion
    Harp Harps Marimba Marimbas Xylophone Xylophones Kalimba Kalimbas
    Glockenspiel Glockenspiels Bells TubularBells Chimes Vibes Vibraphone Vibraphones
    SteelDrum SteelDrums SteelPan SteelPans Cajon Cajons Djembe Djembees

    # Performance techniques / musical articulations
    Muted Mutes Fingered Slap Slaps Slide Slides Bowed Bows
    Harmonics Pizzicato Pizzicatos Tremolo Tremolos Glissando Glissandos
    Vibrato Vibratos Legato Staccato Plucked Strummed
    Riff Riffs Lick Licks Hook Hooks Fill Fills Run Runs

    # Common DAW/sample terms
    Dry Wets Wet Room Rooms Hall Halls Plate Plates
    Delay Delays Reverb Reverbs Chorus Flanger Phaser
    Dist Dists Overdrive Fuzz

    # General musical terms
    Bassline Basslines Harmony Harmonies Chord Chords
    Progression Progressions Scale Scales Mode Modes
    Interval Intervals Timbre Timbres Texture Textures
    Phrase Phrases Motif Motifs
"""
# Filter out comments and create the final list
NEGATIVE_KEYWORDS = [word for word in NEGATIVE_KEYWORDS_TEXT.strip().split() if not word.startswith('#')]
negative_pattern = re.compile(r"|".join(re.escape(word) for word in NEGATIVE_KEYWORDS), re.IGNORECASE)


# --- DURATION CACHE ---
DURATION_CACHE_FILE = Path("duration_cache.json")

def get_duration_cached(file_path, cache):
    """
    Gets audio duration, using a cache to avoid re-calculating.
    """
    file_path_str = str(file_path)
    if file_path_str in cache:
        return cache[file_path_str]
    
    try:
        import torchaudio
        info = torchaudio.info(file_path_str)
        duration = info.num_frames / info.sample_rate
        cache[file_path_str] = duration
        return duration
    except Exception as e:
        tqdm.write(f"⚠️  Warning: Could not get duration for {file_path_str}. Error: {e}")
        return None

def main():
    """
    Main function to read paths, group by session and duration, 
    and identify drum kits.
    """
    if not AUDIO_LIST_FILE.exists():
        print(f"Error: Audio list file not found at {AUDIO_LIST_FILE}")
        return

    # Load caches and existing results
    duration_cache = {}
    if DURATION_CACHE_FILE.exists():
        with open(DURATION_CACHE_FILE, 'r') as f:
            duration_cache = json.load(f)

    final_drum_groups = defaultdict(list)
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r') as f:
            final_drum_groups.update(json.load(f))
            print(f"Loaded {len(final_drum_groups)} sessions from existing results file.")

    # 1. Load all audio paths from the text file
    print(f"🔎 Loading audio paths from {AUDIO_LIST_FILE}...")
    with open(AUDIO_LIST_FILE, 'r') as f:
        all_paths = [Path(line.strip()) for line in f if line.strip()]
    print(f"Found {len(all_paths)} total paths.")

    # 2. Group files by their parent session directory
    print("📁 Grouping files by session directory...")
    session_files = defaultdict(list)
    for path in tqdm(all_paths, desc="Grouping by Session"):
        session_dir = str(path.parent)
        session_files[session_dir].append(path)
    print(f"Found {len(session_files)} unique session directories.")

    # 3. Process each session to find drum groups
    tqdm.write("\n--- Starting Session Processing ---")

    for session_dir, files_in_session in tqdm(session_files.items(), desc="Processing Sessions"):
        
        # Pre-calculate durations for all valid files in the session
        valid_files_with_duration = []
        for file_path in files_in_session:
            duration = get_duration_cached(file_path, duration_cache)
            if duration is not None and duration >= MIN_DURATION_S:
                valid_files_with_duration.append((str(file_path), duration))

        # --- NEW: "ANCHORED SWEEP" CLUSTERING LOGIC ---
        
        # Step 1: Sort the files by duration.
        valid_files_with_duration.sort(key=lambda x: x[1])

        # Step 2: Iterate through the sorted list and form clusters.
        clusters = []
        if not valid_files_with_duration:
            continue

        processed_indices = set()
        for i in range(len(valid_files_with_duration)):
            if i in processed_indices:
                continue

            # This file is the anchor for a new potential cluster
            anchor_path, anchor_dur = valid_files_with_duration[i]
            current_cluster = [anchor_path]
            processed_indices.add(i)

            # Check all subsequent files against this anchor
            for j in range(i + 1, len(valid_files_with_duration)):
                if j in processed_indices:
                    continue
                
                other_path, other_dur = valid_files_with_duration[j]
                
                # If the next file is within tolerance of the anchor, add it
                if (other_dur - anchor_dur) <= DURATION_TOLERANCE:
                    current_cluster.append(other_path)
                    processed_indices.add(j)
                else:
                    # Since the list is sorted, no further files will match this anchor
                    break 
            
            clusters.append(current_cluster)


        # Step 3: Validate each complete cluster.
        for cluster in clusters:
            if len(cluster) < 2:
                continue

            has_drum_track = False
            has_negative_track = False
            for file_path_str in cluster:
                file_name = Path(file_path_str).name
                if negative_pattern.search(file_name):
                    has_negative_track = True
                    break
                if not has_drum_track and drum_pattern.search(file_name):
                    has_drum_track = True
            
            # If the cluster is a valid drum group, add it.
            if has_drum_track and not has_negative_track:
                avg_duration = sum(get_duration_cached(p, duration_cache) for p in cluster) / len(cluster)

                group_data = {
                    'duration_sec': avg_duration,
                    'tracks': sorted(list(cluster))
                }
                
                # Avoid adding duplicate groups if script is rerun
                if group_data not in final_drum_groups.get(session_dir, []):
                    final_drum_groups[session_dir].append(group_data)
                    
                    # Save the entire updated dictionary to the JSON file immediately.
                    with open(OUTPUT_JSON, 'w') as f:
                        json.dump(final_drum_groups, f, indent=4)
                    
                    # Print the newly found group to the console
                    tqdm.write("-" * 50)
                    tqdm.write(f"✅ DRUM GROUP FOUND in Session: {session_dir}")
                    tqdm.write(f"   Duration: ~{avg_duration:.2f} seconds")
                    for track in group_data['tracks']:
                        tqdm.write(f"   - {Path(track).name}")
                    tqdm.write("-" * 50)


    # 4. Final save
    print(f"\n--- Processing Complete ---")
    print(f"✅ Found drum groups in {len(final_drum_groups)} sessions.")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_drum_groups, f, indent=4)
    print(f"Final results saved to {OUTPUT_JSON}")

    with open(DURATION_CACHE_FILE, 'w') as f:
        json.dump(duration_cache, f, indent=4)
    print(f"Duration cache saved to {DURATION_CACHE_FILE}")


if __name__ == "__main__":
    main()
