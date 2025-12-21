
import soundfile as sf
import sounddevice as sd
import numpy as np

# Adjust these if needed
SILENCE_THRESHOLD = 0.001     # amplitude below this = silence
WINDOW_SECONDS = 0.25         # size of scanning window for silence detection
PLAY_SECONDS = 5.0            # duration of audio to play


def find_first_loud_frame(audio, sr, threshold=SILENCE_THRESHOLD, window_sec=WINDOW_SECONDS):
    """
    Returns the sample index where audio first exceeds silence threshold.
    """
    window_size = int(window_sec * sr)
    num_windows = len(audio) // window_size

    for w in range(num_windows):
        seg = audio[w * window_size : (w + 1) * window_size]

        # Check RMS of window
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > threshold:
            return w * window_size

    # fallback: start at 0
    return 0


def play_5_seconds(path):
    audio, sr = sf.read(path, dtype='float32')

    # mono/stereo fix
    if audio.ndim > 1:
        audio = audio[:, 0]   # take first channel

    start = find_first_loud_frame(audio, sr)
    end = start + int(PLAY_SECONDS * sr)

    # Clamp end index
    end = min(end, len(audio))

    clip = audio[start:end]

    print(f"\nPlaying {path}")
    print(f"  → Start sample: {start}, seconds: {start/sr:.2f}")
    print(f"  → Playing {len(clip)/sr:.2f} sec")

    sd.play(clip, sr)
    sd.wait()



paths = [
    "/home/arlo/gcs-bucket/protools/2025-04-02/New/Joav .04/Audio Files/414 trumpet.02_05.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-02/New/Joav 00/Audio Files/414 trumpet.03_06.wav",

    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_15.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_16.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_18.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_36.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_38.wav",
    "/home/arlo/gcs-bucket/protools/2025-04-15/New/2024.08.02_StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_40.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-11/ArloStems/BRASS/TPT/TPTMUTE.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-11/ArloStems/BRASS/TPT/TPTMUTE_1.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-11/ArloStems/BRASS/TPT/TPTMUTE_2.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpet 3.01_02.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpet 3_01.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpet.29_45.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpet.30_46.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2.01_07.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2.02_08.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2.35_83.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2.36_84.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2_01.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2_03.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2_04.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/Trumpets 2_05.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/trumpet 4.01_02.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/trumpet 4.02_03.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.05.21_RyotaSasaki_Friends/Audio Files/trumpet 4_01.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.4.30_UnderTheSakuraSky_48bit/Audio Files/Trumpet2 - Yunho.01_04.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.4.30_UnderTheSakuraSky_48bit/Audio Files/trumpet 3.05_11.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-05-29/New/2025.4.30_UnderTheSakuraSky_48bit/Audio Files/trumpet 3.07_13.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.02_02.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.04_04.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.09_09.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.10_10.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.12_14.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.13_15.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.15_17.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.16_18.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.25_30.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.26_31.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.28_32.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.37_40.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.38_41.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-06-15/New/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.39_42.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-10-05/Prev/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.05_05.wav",
    "/home/arlo/gcs-bucket/protoolsA/2025-10-05/Prev/Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.07_07.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-10-06/Prev/2025.09.27_MP340_Experimental2_JadeFaria/Audio Files/Trumpet(2).00_02.wav",

    "/home/arlo/gcs-bucket/protoolsA/2025-10-12/Prev/DevonGates_DaveThePotter/Audio Files/Trumpet.03_10.wav"
]


for p in paths:
    play_5_seconds(p)

