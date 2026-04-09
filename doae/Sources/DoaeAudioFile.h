#pragma once
#define __USE_PUBLIC_HEADERS__ 1
#include "DoaeFileFormat.h"
#include "DoaeDecoder.h"
#include <AudioToolbox/AudioFile.h>
#include <AudioToolbox/AudioFileComponent.h>
#include <vector>
#include <string>
#include <memory>

typedef UInt32 AudioFileComponentPropertyID;

// Per-instance state for an open .doae file.
// Decodes all stems to a single stereo premix on open.
// Subsequent ReadPackets calls serve from the PCM cache.

class DoaeAudioFile {
public:
    DoaeAudioFile();
    ~DoaeAudioFile();

    // Open the file at path, decode latents → premixed stereo PCM.
    // Returns noErr on success, an OSStatus error code otherwise.
    OSStatus open(CFURLRef url, SInt8 permissions, const std::string& modelPath);

    void close();

    // AudioFileComponent selectors
    OSStatus getPropertyInfo(AudioFileComponentPropertyID propID,
                             UInt32* outSize, UInt32* outWritable);
    OSStatus getProperty(AudioFileComponentPropertyID propID,
                         UInt32* ioSize, void* outData);
    OSStatus readPackets(Boolean useCache,
                         UInt32* outNumBytes,
                         AudioStreamPacketDescription* outPD,
                         SInt64 inStartPacket,
                         UInt32* ioNumPackets,
                         void* outData);
    OSStatus readPacketData(Boolean useCache,
                            UInt32* ioNumBytes,
                            AudioStreamPacketDescription* outPD,
                            SInt64 inStartPacket,
                            UInt32* ioNumPackets,
                            void* outData);

    // Static: return file type info without an open file
    static OSStatus extensionIsThisFormat(CFStringRef ext, UInt32* outResult);
    static OSStatus fileIsThisFormat(SInt16 fd, SInt64 fileOffset,
                                     UInt32 headerSize, const void* header,
                                     UInt32* outResult);
    static OSStatus getGlobalInfoSize(AudioFileComponentPropertyID propID,
                                      UInt32 specSize, const void* spec,
                                      UInt32* outSize);
    static OSStatus getGlobalInfo(AudioFileComponentPropertyID propID,
                                  UInt32 specSize, const void* spec,
                                  UInt32* ioSize, void* outData);

private:
    OSStatus decodeFile(const char* path, const std::string& modelPath);

    // Decoded PCM: stereo float32, non-interleaved: [left[N], right[N]]
    std::vector<float> m_pcm;   // size = 2 * m_totalFrames
    uint64_t           m_totalFrames = 0;
    uint32_t           m_sampleRate  = DOAE_SAMPLE_RATE;
    uint32_t           m_channels    = 2;
    bool               m_open        = false;

    // Format description for Core Audio
    AudioStreamBasicDescription m_asbd = {};
};
