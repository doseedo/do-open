#include "DoaeAudioFile.h"
#include <CoreFoundation/CoreFoundation.h>
#include <syslog.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>

DoaeAudioFile::DoaeAudioFile() {}
DoaeAudioFile::~DoaeAudioFile() { close(); }

void DoaeAudioFile::close() {
    m_pcm.clear();
    m_totalFrames = 0;
    m_open = false;
}

// ---------------------------------------------------------------------------
// Open + decode
// ---------------------------------------------------------------------------

OSStatus DoaeAudioFile::open(CFURLRef url, SInt8 /*permissions*/,
                             const std::string& modelPath) {
    char path[PATH_MAX];
    if (!CFURLGetFileSystemRepresentation(url, true,
            reinterpret_cast<UInt8*>(path), sizeof(path))) {
        return kAudioFileInvalidFileError;
    }
    return decodeFile(path, modelPath);
}

OSStatus DoaeAudioFile::decodeFile(const char* path, const std::string& modelPath) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) return kAudioFilePermissionsError;

    // --- Read header ---
    DoaeHeader hdr;
    if (::read(fd, &hdr, sizeof(hdr)) != sizeof(hdr)) {
        ::close(fd); return kAudioFileInvalidFileError;
    }
    if (hdr.magic != DOAE_MAGIC) {
        ::close(fd); return kAudioFileInvalidFileError;
    }
    if (hdr.version != DOAE_VERSION) {
        ::close(fd); return kAudioFileUnsupportedFileTypeError;
    }

    // --- Read stem info blocks ---
    uint16_t stemCount = hdr.stem_count;
    std::vector<DoaeStemInfo> stems(stemCount);
    for (int i = 0; i < stemCount; i++) {
        if (::read(fd, &stems[i], sizeof(DoaeStemInfo)) != sizeof(DoaeStemInfo)) {
            ::close(fd); return kAudioFileInvalidFileError;
        }
    }

    // --- Compute total output frames (from first non-muted stem) ---
    uint64_t totalFrames = 0;
    for (auto& s : stems) {
        if (!s.muted) {
            totalFrames = doae_audio_samples(s.latent_frames);
            break;
        }
    }
    if (totalFrames == 0) { ::close(fd); return kAudioFileBadPropertySizeError; }

    // --- Allocate premix buffer: stereo float32 [2][totalFrames] ---
    std::vector<float> premix(2 * totalFrames, 0.0f);

    // --- Load decoder ---
    OobleckDecoder decoder(modelPath);
    if (!decoder.isReady()) {
        syslog(LOG_ERR, "DoaeCodec: decoder init failed: %s",
               decoder.lastError().c_str());
        ::close(fd);
        return kAudioFileUnsupportedDataFormatError;
    }

    // --- Decode each stem and mix into premix ---
    // We decode in chunks of T=50 (2s) to limit peak memory usage
    static const int CHUNK_T = 50;

    for (auto& stem : stems) {
        if (stem.muted) continue;

        float stemGain = stem.gain;
        float panL = std::min(1.0f, 1.0f - stem.pan);
        float panR = std::min(1.0f, 1.0f + stem.pan);
        float gainL = stemGain * panL;
        float gainR = stemGain * panR;

        uint32_t T = stem.latent_frames;
        uint32_t latentDim = stem.latent_dim ? stem.latent_dim : 64;
        uint64_t latentBytes = (uint64_t)T * latentDim * sizeof(float);

        // Read the full latent for this stem into memory
        std::vector<float> latentBuf(T * latentDim);
        if (::lseek(fd, stem.latent_offset, SEEK_SET) < 0) {
            ::close(fd); return kAudioFileInvalidFileError;
        }
        ssize_t toRead = latentBytes;
        ssize_t got = ::read(fd, latentBuf.data(), toRead);
        if (got != toRead) {
            ::close(fd); return kAudioFileInvalidFileError;
        }

        // Decode in CHUNK_T-frame chunks
        std::vector<float> chunkPCM;
        uint32_t frameOffset = 0;
        while (frameOffset < T) {
            uint32_t chunkT = std::min((uint32_t)CHUNK_T, T - frameOffset);
            uint64_t audioOffset = (uint64_t)frameOffset * DOAE_DOWNSAMPLE;

            // latentBuf layout: [64][T] row-major → chunk is [64][chunkT]
            // We need [64 * chunkT] in column-major order per the ONNX model
            // The stored layout is channel-first: latentBuf[ch * T + frame]
            std::vector<float> chunkLatent(latentDim * chunkT);
            for (uint32_t ch = 0; ch < latentDim; ch++) {
                for (uint32_t f = 0; f < chunkT; f++) {
                    chunkLatent[ch * chunkT + f] = latentBuf[ch * T + (frameOffset + f)];
                }
            }

            if (!decoder.decode(chunkLatent.data(), chunkT, chunkPCM)) {
                syslog(LOG_ERR, "DoaeCodec: decode chunk failed at frame %u", frameOffset);
                ::close(fd); return kAudioFileUnsupportedDataFormatError;
            }

            // chunkPCM: [left[N], right[N]] where N = chunkT * DOAE_DOWNSAMPLE
            uint64_t N = (uint64_t)chunkT * DOAE_DOWNSAMPLE;
            float* leftOut  = premix.data() + audioOffset;
            float* rightOut = premix.data() + totalFrames + audioOffset;
            const float* leftSrc  = chunkPCM.data();
            const float* rightSrc = chunkPCM.data() + N;

            for (uint64_t i = 0; i < N && (audioOffset + i) < totalFrames; i++) {
                leftOut[i]  += leftSrc[i]  * gainL;
                rightOut[i] += rightSrc[i] * gainR;
            }

            frameOffset += chunkT;
        }
    }

    ::close(fd);

    m_pcm         = std::move(premix);
    m_totalFrames = totalFrames;
    m_sampleRate  = hdr.sample_rate ? hdr.sample_rate : DOAE_SAMPLE_RATE;
    m_channels    = 2;
    m_open        = true;

    // Build ASBD for non-interleaved float32 PCM
    memset(&m_asbd, 0, sizeof(m_asbd));
    m_asbd.mSampleRate       = m_sampleRate;
    m_asbd.mFormatID         = kAudioFormatLinearPCM;
    m_asbd.mFormatFlags      = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
                               | kAudioFormatFlagIsNonInterleaved;
    m_asbd.mBytesPerPacket   = sizeof(float);
    m_asbd.mFramesPerPacket  = 1;
    m_asbd.mBytesPerFrame    = sizeof(float);
    m_asbd.mChannelsPerFrame = m_channels;
    m_asbd.mBitsPerChannel   = 32;

    syslog(LOG_INFO, "DoaeCodec: opened %s → %llu frames @ %u Hz",
           path, (unsigned long long)m_totalFrames, m_sampleRate);
    return noErr;
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

OSStatus DoaeAudioFile::getPropertyInfo(AudioFileComponentPropertyID propID,
                                        UInt32* outSize, UInt32* outWritable) {
    if (outWritable) *outWritable = false;
    switch (propID) {
        case kAudioFilePropertyFileFormat:
            if (outSize) *outSize = sizeof(AudioFileTypeID); return noErr;
        case kAudioFilePropertyDataFormat:
            if (outSize) *outSize = sizeof(AudioStreamBasicDescription); return noErr;
        case kAudioFilePropertyAudioDataByteCount:
            if (outSize) *outSize = sizeof(SInt64); return noErr;
        case kAudioFilePropertyAudioDataPacketCount:
            if (outSize) *outSize = sizeof(SInt64); return noErr;
        case kAudioFilePropertyIsOptimized:
            if (outSize) *outSize = sizeof(UInt32); return noErr;
        case kAudioFilePropertyPacketSizeUpperBound:
            if (outSize) *outSize = sizeof(UInt32); return noErr;
        case kAudioFilePropertyMaximumPacketSize:
            if (outSize) *outSize = sizeof(UInt32); return noErr;
        default:
            return kAudioFileUnsupportedPropertyError;
    }
}

OSStatus DoaeAudioFile::getProperty(AudioFileComponentPropertyID propID,
                                    UInt32* ioSize, void* outData) {
    switch (propID) {
        case kAudioFilePropertyFileFormat: {
            if (*ioSize < sizeof(AudioFileTypeID)) return kAudioFileBadPropertySizeError;
            AudioFileTypeID t = 'DOAE';
            memcpy(outData, &t, sizeof(t));
            *ioSize = sizeof(t);
            return noErr;
        }
        case kAudioFilePropertyDataFormat: {
            if (*ioSize < sizeof(m_asbd)) return kAudioFileBadPropertySizeError;
            memcpy(outData, &m_asbd, sizeof(m_asbd));
            *ioSize = sizeof(m_asbd);
            return noErr;
        }
        case kAudioFilePropertyAudioDataByteCount: {
            SInt64 v = (SInt64)(m_totalFrames * m_channels * sizeof(float));
            memcpy(outData, &v, sizeof(v)); *ioSize = sizeof(v); return noErr;
        }
        case kAudioFilePropertyAudioDataPacketCount: {
            SInt64 v = (SInt64)m_totalFrames;
            memcpy(outData, &v, sizeof(v)); *ioSize = sizeof(v); return noErr;
        }
        case kAudioFilePropertyIsOptimized: {
            UInt32 v = 1;
            memcpy(outData, &v, sizeof(v)); *ioSize = sizeof(v); return noErr;
        }
        case kAudioFilePropertyPacketSizeUpperBound:
        case kAudioFilePropertyMaximumPacketSize: {
            UInt32 v = (UInt32)(m_channels * sizeof(float));
            memcpy(outData, &v, sizeof(v)); *ioSize = sizeof(v); return noErr;
        }
        default:
            return kAudioFileUnsupportedPropertyError;
    }
}

// ---------------------------------------------------------------------------
// Read audio data
// ---------------------------------------------------------------------------

OSStatus DoaeAudioFile::readPackets(Boolean /*useCache*/,
                                    UInt32* outNumBytes,
                                    AudioStreamPacketDescription* /*outPD*/,
                                    SInt64 inStartPacket,
                                    UInt32* ioNumPackets,
                                    void* outData) {
    if (!m_open || !outData) return kAudioFileInvalidFileError;

    // For LPCM, 1 packet = 1 frame
    SInt64 startFrame = inStartPacket;
    UInt32 reqPackets = *ioNumPackets;

    if (startFrame >= (SInt64)m_totalFrames) {
        *ioNumPackets = 0;
        if (outNumBytes) *outNumBytes = 0;
        return kAudioFileEndOfFileError;
    }

    UInt32 available = (UInt32)(m_totalFrames - startFrame);
    UInt32 toRead    = std::min(reqPackets, available);

    // Non-interleaved output: buffer layout per channel
    // outData is a single buffer; for non-interleaved, Core Audio provides
    // an AudioBufferList, but the legacy ReadPackets signature gives a flat void*.
    // We output interleaved here for compatibility with the flat buffer API.
    float* dst = reinterpret_cast<float*>(outData);
    const float* srcL = m_pcm.data() + startFrame;
    const float* srcR = m_pcm.data() + m_totalFrames + startFrame;

    for (UInt32 i = 0; i < toRead; i++) {
        dst[i * 2]     = srcL[i];
        dst[i * 2 + 1] = srcR[i];
    }

    *ioNumPackets = toRead;
    if (outNumBytes) *outNumBytes = toRead * 2 * sizeof(float);
    return toRead < reqPackets ? kAudioFileEndOfFileError : noErr;
}

OSStatus DoaeAudioFile::readPacketData(Boolean useCache,
                                       UInt32* ioNumBytes,
                                       AudioStreamPacketDescription* outPD,
                                       SInt64 inStartPacket,
                                       UInt32* ioNumPackets,
                                       void* outData) {
    return readPackets(useCache, ioNumBytes, outPD, inStartPacket, ioNumPackets, outData);
}

// ---------------------------------------------------------------------------
// Static: file type probing (no open file)
// ---------------------------------------------------------------------------

OSStatus DoaeAudioFile::extensionIsThisFormat(CFStringRef ext, UInt32* outResult) {
    *outResult = 0;
    if (ext && CFStringCompare(ext, CFSTR("doae"), kCFCompareCaseInsensitive) == kCFCompareEqualTo) {
        *outResult = 1;
    }
    return noErr;
}

OSStatus DoaeAudioFile::fileIsThisFormat(SInt16 fd, SInt64 /*fileOffset*/,
                                         UInt32 headerSize, const void* header,
                                         UInt32* outResult) {
    *outResult = 0;
    if (headerSize >= 4 && header) {
        uint32_t magic;
        memcpy(&magic, header, 4);
        if (magic == DOAE_MAGIC) *outResult = 100;  // confidence 100%
    }
    return noErr;
}

OSStatus DoaeAudioFile::getGlobalInfoSize(AudioFileComponentPropertyID propID,
                                          UInt32 /*specSize*/, const void* /*spec*/,
                                          UInt32* outSize) {
    switch (propID) {
        case kAudioFileComponent_CanRead:
        case kAudioFileComponent_CanWrite:
            *outSize = sizeof(UInt32); return noErr;
        case kAudioFileComponent_FileTypeName:
            *outSize = sizeof(CFStringRef); return noErr;
        case kAudioFileComponent_ExtensionsForType:
        case kAudioFileComponent_UTIsForType:
        case kAudioFileComponent_MIMETypesForType:
        case kAudioFileComponent_AvailableFormatIDs:
            *outSize = sizeof(CFArrayRef); return noErr;
        default:
            return kAudioFileUnsupportedPropertyError;
    }
}

OSStatus DoaeAudioFile::getGlobalInfo(AudioFileComponentPropertyID propID,
                                      UInt32 /*specSize*/, const void* /*spec*/,
                                      UInt32* ioSize, void* outData) {
    switch (propID) {
        case kAudioFileComponent_CanRead: {
            UInt32 v = 1; memcpy(outData, &v, sizeof(v)); return noErr;
        }
        case kAudioFileComponent_CanWrite: {
            UInt32 v = 0; memcpy(outData, &v, sizeof(v)); return noErr;
        }
        case kAudioFileComponent_FileTypeName: {
            CFStringRef s = CFSTR("Doseedo Audio Engine");
            *(CFStringRef*)outData = (CFStringRef)CFRetain(s);
            return noErr;
        }
        case kAudioFileComponent_ExtensionsForType: {
            CFStringRef ext = CFSTR("doae");
            CFArrayRef arr = CFArrayCreate(NULL, (const void**)&ext, 1, &kCFTypeArrayCallBacks);
            *(CFArrayRef*)outData = arr;
            return noErr;
        }
        case kAudioFileComponent_UTIsForType: {
            CFStringRef uti = CFSTR("com.doseedo.doae");
            CFArrayRef arr = CFArrayCreate(NULL, (const void**)&uti, 1, &kCFTypeArrayCallBacks);
            *(CFArrayRef*)outData = arr;
            return noErr;
        }
        case kAudioFileComponent_MIMETypesForType: {
            CFStringRef mime = CFSTR("audio/x-doae");
            CFArrayRef arr = CFArrayCreate(NULL, (const void**)&mime, 1, &kCFTypeArrayCallBacks);
            *(CFArrayRef*)outData = arr;
            return noErr;
        }
        case kAudioFileComponent_AvailableFormatIDs: {
            AudioFormatID fmt = kAudioFormatLinearPCM;
            CFNumberRef n = CFNumberCreate(NULL, kCFNumberSInt32Type, &fmt);
            CFArrayRef arr = CFArrayCreate(NULL, (const void**)&n, 1, &kCFTypeArrayCallBacks);
            CFRelease(n);
            *(CFArrayRef*)outData = arr;
            return noErr;
        }
        default:
            return kAudioFileUnsupportedPropertyError;
    }
}
