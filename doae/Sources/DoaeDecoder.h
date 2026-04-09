#pragma once
#include <vector>
#include <string>
#include <cstdint>

// OobleckDecoder: wraps ONNX Runtime to decode latent [1,64,T] → PCM float32 [2,N]
// Thread-safe: each instance owns its own OrtSession.

class OobleckDecoder {
public:
    // modelPath: path to oobleck_decoder_packed.onnx
    OobleckDecoder(const std::string& modelPath);
    ~OobleckDecoder();

    // Decode a latent block. latent is float32[64 * T] in channel-first order.
    // Returns interleaved stereo: result[0..N-1] = left, result[N..2N-1] = right
    // where N = T * 1920.
    // Returns false on error.
    bool decode(const float* latent, int T, std::vector<float>& pcmOut);

    bool isReady() const { return m_ready; }
    std::string lastError() const { return m_lastError; }

private:
    struct Impl;
    Impl* m_impl = nullptr;
    bool  m_ready = false;
    std::string m_lastError;
};
