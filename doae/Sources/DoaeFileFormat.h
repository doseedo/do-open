#pragma once
#include <stdint.h>

// .doae — Doseedo Audio Engine file format
// Magic bytes: "DOAE" (little-endian uint32 = 0x45414F44)

#define DOAE_MAGIC       0x45414F44u   // "DOAE"
#define DOAE_VERSION     1
#define DOAE_SAMPLE_RATE 48000u
#define DOAE_LATENT_DIM  64            // latent channels (Oobleck encoder dim)
#define DOAE_DOWNSAMPLE  1920          // audio samples per latent frame (2*4*4*6*10)

#pragma pack(push, 1)

typedef struct {
    uint32_t magic;          // DOAE_MAGIC
    uint16_t version;        // DOAE_VERSION
    uint16_t stem_count;     // number of stems in this file
    uint32_t sample_rate;    // = 48000
    uint32_t reserved[4];    // zeroed, reserved for future use
} DoaeHeader;                // 28 bytes

typedef struct {
    char     name[64];       // stem name, null-terminated and null-padded
    float    gain;           // linear gain (default 1.0)
    float    pan;            // -1.0 (L) to +1.0 (R), default 0.0
    uint8_t  muted;          // 0 = active, 1 = muted
    uint8_t  reserved[3];    // zeroed
    uint64_t latent_offset;  // byte offset from start of file to latent data
    uint32_t latent_frames;  // T: number of latent time frames
    uint32_t latent_dim;     // = 64 (latent channels)
} DoaeStemInfo;              // 96 bytes

#pragma pack(pop)

// Latent data layout (per stem):
//   float32[latent_dim][latent_frames]   row-major, C contiguous
//   i.e. float32[64][T] stored as T*64 floats, channel-first
//   Total bytes per stem: latent_frames * latent_dim * sizeof(float)
//
// Audio output per stem (decoded):
//   float32[2][N] where N = latent_frames * DOAE_DOWNSAMPLE
//   stereo, 48kHz

// Utility: audio sample count from latent frames
static inline uint64_t doae_audio_samples(uint32_t latent_frames) {
    return (uint64_t)latent_frames * DOAE_DOWNSAMPLE;
}
