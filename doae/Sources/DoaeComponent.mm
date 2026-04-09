// DoaeComponent.mm
// Core Audio File Component — selector dispatch via AudioComponentPlugInInterface::Lookup.
// Each AudioFileComponent selector maps to a static C function with the exact
// signature defined in AudioFileComponent.h.

#define __USE_PUBLIC_HEADERS__ 1
#include "DoaeAudioFile.h"
#include <AudioToolbox/AudioToolbox.h>
#include <AudioToolbox/AudioFileComponent.h>
#include <CoreFoundation/CoreFoundation.h>
#include <syslog.h>
#include <string>

// ---------------------------------------------------------------------------
// Per-instance struct
// ---------------------------------------------------------------------------

struct DoaeInstance {
    AudioComponentPlugInInterface plugIn;   // MUST be first
    AudioComponentInstance        acInst;
    DoaeAudioFile*                audioFile;
    std::string                   modelPath;
};

static std::string modelPathInBundle() {
    CFBundleRef bundle = CFBundleGetBundleWithIdentifier(CFSTR("com.doseedo.DoaeCodec"));
    if (!bundle) {
        return std::string(getenv("HOME") ?: "") +
               "/Library/Audio/Plug-Ins/Components/DoaeCodec.component"
               "/Contents/Resources/oobleck_decoder_packed.onnx";
    }
    CFURLRef url = CFBundleCopyResourceURL(bundle, CFSTR("oobleck_decoder_packed"),
                                           CFSTR("onnx"), NULL);
    if (!url) return "";
    char path[PATH_MAX];
    CFURLGetFileSystemRepresentation(url, true, (UInt8*)path, sizeof(path));
    CFRelease(url);
    return std::string(path);
}

// ---------------------------------------------------------------------------
// Per-selector static C functions (must match AudioFileComponent Proc typedefs)
// ---------------------------------------------------------------------------

static OSStatus DoaeOpenURL(void* self, CFURLRef url, SInt8 perm, int /*fd*/) {
    auto* inst = (DoaeInstance*)self;
    return inst->audioFile->open(url, perm, inst->modelPath);
}

static OSStatus DoaeClose(void* self) {
    auto* inst = (DoaeInstance*)self;
    if (inst->audioFile) inst->audioFile->close();
    return noErr;
}

static OSStatus DoaeReadPackets(void* self, Boolean useCache,
                                UInt32* outNumBytes,
                                AudioStreamPacketDescription* outPD,
                                SInt64 startPacket, UInt32* ioNumPackets,
                                void* outBuffer) {
    auto* inst = (DoaeInstance*)self;
    return inst->audioFile->readPackets(useCache, outNumBytes, outPD,
                                        startPacket, ioNumPackets, outBuffer);
}

static OSStatus DoaeReadPacketData(void* self, Boolean useCache,
                                   UInt32* ioNumBytes,
                                   AudioStreamPacketDescription* outPD,
                                   SInt64 startPacket, UInt32* ioNumPackets,
                                   void* outBuffer) {
    auto* inst = (DoaeInstance*)self;
    return inst->audioFile->readPacketData(useCache, ioNumBytes, outPD,
                                           startPacket, ioNumPackets, outBuffer);
}

static OSStatus DoaeGetPropertyInfo(void* self, AudioFileComponentPropertyID propID,
                                    UInt32* outSize, UInt32* outWritable) {
    auto* inst = (DoaeInstance*)self;
    return inst->audioFile->getPropertyInfo(propID, outSize, outWritable);
}

static OSStatus DoaeGetProperty(void* self, AudioFileComponentPropertyID propID,
                                UInt32* ioSize, void* outData) {
    auto* inst = (DoaeInstance*)self;
    return inst->audioFile->getProperty(propID, ioSize, outData);
}

// File-type probing selectors (called on an instance with no open file)
static OSStatus DoaeExtensionIsThisFormat(void* /*self*/, CFStringRef ext, UInt32* outResult) {
    return DoaeAudioFile::extensionIsThisFormat(ext, outResult);
}

static OSStatus DoaeFileDataIsThisFormat(void* /*self*/, UInt32 dataSize,
                                         const void* data, UInt32* outResult) {
    *outResult = 0;
    if (dataSize >= 4 && data) {
        uint32_t magic; memcpy(&magic, data, 4);
        if (magic == DOAE_MAGIC) *outResult = 100;
    }
    return noErr;
}

static OSStatus DoaeGetGlobalInfoSize(void* /*self*/, AudioFileComponentPropertyID propID,
                                      UInt32 specSize, const void* spec, UInt32* outSize) {
    return DoaeAudioFile::getGlobalInfoSize(propID, specSize, spec, outSize);
}

static OSStatus DoaeGetGlobalInfo(void* /*self*/, AudioFileComponentPropertyID propID,
                                  UInt32 specSize, const void* spec,
                                  UInt32* ioSize, void* outData) {
    return DoaeAudioFile::getGlobalInfo(propID, specSize, spec, ioSize, outData);
}

// ---------------------------------------------------------------------------
// Lookup — maps selectors to function pointers
// ---------------------------------------------------------------------------

static AudioComponentMethod DoaeLookup(SInt16 selector) {
    switch (selector) {
        case kAudioFileOpenURLSelect:                return (AudioComponentMethod)DoaeOpenURL;
        case kAudioFileCloseSelect:                  return (AudioComponentMethod)DoaeClose;
        case kAudioFileReadPacketsSelect:            return (AudioComponentMethod)DoaeReadPackets;
        case kAudioFileReadPacketDataSelect:         return (AudioComponentMethod)DoaeReadPacketData;
        case kAudioFileGetPropertyInfoSelect:        return (AudioComponentMethod)DoaeGetPropertyInfo;
        case kAudioFileGetPropertySelect:            return (AudioComponentMethod)DoaeGetProperty;
        case kAudioFileExtensionIsThisFormatSelect:  return (AudioComponentMethod)DoaeExtensionIsThisFormat;
        case kAudioFileFileDataIsThisFormatSelect:   return (AudioComponentMethod)DoaeFileDataIsThisFormat;
        case kAudioFileGetGlobalInfoSizeSelect:      return (AudioComponentMethod)DoaeGetGlobalInfoSize;
        case kAudioFileGetGlobalInfoSelect:          return (AudioComponentMethod)DoaeGetGlobalInfo;
        default: return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Open / Close (instance lifecycle, called by AudioComponentInstanceNew/Dispose)
// ---------------------------------------------------------------------------

static OSStatus DoaePlugInOpen(void* self, AudioComponentInstance inst) {
    auto* me  = (DoaeInstance*)self;
    me->acInst    = inst;
    me->audioFile = new DoaeAudioFile();
    me->modelPath = modelPathInBundle();
    syslog(LOG_INFO, "DoaeCodec: component opened. model=%s", me->modelPath.c_str());
    return noErr;
}

static OSStatus DoaePlugInClose(void* self) {
    auto* me = (DoaeInstance*)self;
    delete me->audioFile;
    me->audioFile = nullptr;
    return noErr;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

extern "C" __attribute__((visibility("default")))
AudioComponentPlugInInterface* DoaeComponentFactory(const AudioComponentDescription* /*desc*/)
{
    auto* inst = new DoaeInstance();
    memset(inst, 0, sizeof(*inst));

    inst->plugIn.Open     = DoaePlugInOpen;
    inst->plugIn.Close    = DoaePlugInClose;
    inst->plugIn.Lookup   = DoaeLookup;
    inst->plugIn.reserved = nullptr;

    return &inst->plugIn;
}
