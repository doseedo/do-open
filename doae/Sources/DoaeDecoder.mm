#include "DoaeDecoder.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <syslog.h>

static Ort::Env& getEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "DoaeCodec");
    return env;
}

struct OobleckDecoder::Impl {
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memInfo;

    Impl() : memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
};

OobleckDecoder::OobleckDecoder(const std::string& modelPath) {
    m_impl = new Impl();

    m_impl->options.SetIntraOpNumThreads(4);
    m_impl->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // CPU-only for in-process component use.
    // CoreML EP could be added here if the host grants GPU access,
    // but for a prototype CPU decode at 13x realtime is sufficient.

    try {
        m_impl->session = std::make_unique<Ort::Session>(
            getEnv(), modelPath.c_str(), m_impl->options);
        m_ready = true;
        syslog(LOG_INFO, "DoaeCodec: decoder ready (%s)", modelPath.c_str());
    } catch (const Ort::Exception& e) {
        m_lastError = e.what();
        syslog(LOG_ERR, "DoaeCodec: session create failed: %s", e.what());
    }
}

OobleckDecoder::~OobleckDecoder() {
    delete m_impl;
}

bool OobleckDecoder::decode(const float* latent, int T, std::vector<float>& pcmOut) {
    if (!m_ready || !m_impl->session) return false;

    // Input tensor shape: [1, 64, T]
    std::array<int64_t, 3> inputShape = {1, 64, (int64_t)T};
    size_t inputElems = 64 * T;

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        m_impl->memInfo,
        const_cast<float*>(latent),
        inputElems,
        inputShape.data(),
        inputShape.size());

    const char* inputNames[]  = {"latent"};
    const char* outputNames[] = {"audio"};

    try {
        auto outputs = m_impl->session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 1);

        auto& outTensor = outputs[0];
        auto shape = outTensor.GetTensorTypeAndShapeInfo().GetShape();
        // shape: [1, 2, N]
        int64_t N = shape[2];
        const float* data = outTensor.GetTensorMutableData<float>();
        // data layout: [left_0..left_N-1, right_0..right_N-1]
        pcmOut.assign(data, data + 2 * N);
        return true;
    } catch (const Ort::Exception& e) {
        m_lastError = e.what();
        syslog(LOG_ERR, "DoaeCodec: decode error: %s", e.what());
        return false;
    }
}
