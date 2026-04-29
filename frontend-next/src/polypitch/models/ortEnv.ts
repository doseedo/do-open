/**
 * ortEnv — centralised configuration for onnxruntime-web.
 *
 * ORT-Web has process-global `env` state (wasm paths, feature flags, thread
 * count, logging). Touching it from multiple model wrappers without a single
 * choke-point leads to race conditions where the second caller's settings are
 * ignored because the WASM backend has already initialised. `configureOrtEnv`
 * is idempotent and runs once on first model load.
 *
 * This file runs inside a Web Worker. It MUST NOT touch `window`, `document`,
 * or anything else that lives on the main thread.
 */

import { env } from "onnxruntime-web";

/** Preferred execution-provider order. WebGPU first when available. */
export type ExecutionProvider = "webgpu" | "wasm" | "cpu";
export const DEFAULT_EXECUTION_PROVIDERS: ExecutionProvider[] = ["wasm"];

let configured = false;
let webgpuAvailable: boolean | null = null;

/**
 * Configure the ORT-Web global env. Safe to call many times — the first call
 * wins and subsequent calls are no-ops.
 *
 * We point `env.wasm.wasmPaths` at a CDN so Vite doesn't have to bundle the
 * .wasm binaries. If you want fully self-hosted builds, override `wasmPaths`
 * before the first model loads.
 */
export function configureOrtEnv(): void {
  if (configured) return;
  configured = true;

  // Serve the WASM binaries from our own /ort/ path. These are copied out of
  // node_modules/onnxruntime-web/dist/ at build time (see scripts/sync-ort-wasm.sh
  // and the copy step in fetch-models). Pinning to a hardcoded CDN version risks
  // drift against the installed JS package — the npm 1.24+ bundle calls internal
  // symbols (e.g. `getValue`) that don't exist in older WASM builds.
  env.wasm.wasmPaths = "/ort/";

  // Pick a sensible thread count. `navigator.hardwareConcurrency` is available
  // in Web Workers; cap at 4 since more threads rarely helps small models and
  // cross-origin isolation is required above 1 thread.
  const hw =
    typeof navigator !== "undefined" && typeof navigator.hardwareConcurrency === "number"
      ? navigator.hardwareConcurrency
      : 4;
  env.wasm.numThreads = Math.min(4, Math.max(1, hw));

  // SIMD + proxy are both good defaults.
  env.wasm.simd = true;
  // env.wasm.proxy can only be used from the main thread. We're in a Worker
  // already, so leave it off.

  // Keep ORT's internal logging quiet in production.
  env.logLevel = process.env.NODE_ENV === "development" ? "verbose" : "error";
}

/**
 * Detect WebGPU availability inside the current (Worker) context. Result is
 * cached; call repeatedly without cost.
 */
export async function detectWebGpu(): Promise<boolean> {
  if (webgpuAvailable !== null) return webgpuAvailable;

  // `navigator.gpu` may be undefined in older Safari or when `--disable-webgpu`
  // is set. We also gate on `requestAdapter()` returning a non-null adapter so
  // software-only stubs don't get picked.
  try {
    const nav = navigator as Navigator & { gpu?: GPU };
    if (!nav.gpu) {
      webgpuAvailable = false;
      return false;
    }
    const adapter = await nav.gpu.requestAdapter();
    webgpuAvailable = adapter !== null;
    return webgpuAvailable;
  } catch {
    webgpuAvailable = false;
    return false;
  }
}

/**
 * Resolve the requested EP list against what the runtime actually supports.
 * Falls back to WASM with a console.warn if WebGPU was requested but absent.
 *
 * Returns the list in the format ORT's `InferenceSession.create` accepts.
 */
export async function resolveExecutionProviders(
  requested: ExecutionProvider[] | undefined,
): Promise<ExecutionProvider[]> {
  const wanted = requested && requested.length > 0 ? requested : DEFAULT_EXECUTION_PROVIDERS;
  const resolved: ExecutionProvider[] = [];
  let warnedWebGpu = false;

  for (const ep of wanted) {
    if (ep === "webgpu") {
      const ok = await detectWebGpu();
      if (ok) {
        resolved.push("webgpu");
      } else if (!warnedWebGpu) {
        warnedWebGpu = true;
        if (process.env.NODE_ENV === 'development') {
          // eslint-disable-next-line no-console
          console.warn(
            "[ortEnv] WebGPU unavailable in this context; falling back to WASM execution provider.",
          );
        }
      }
    } else {
      resolved.push(ep);
    }
  }

  // Always guarantee a terminal fallback so `InferenceSession.create` never
  // receives an empty provider list.
  if (resolved.length === 0) resolved.push("wasm");
  return resolved;
}
