/**
 * Public entrypoint for the ref-renderer pipeline.
 *
 * Pipeline:
 *   1. User provides a brief (text) + optional reference-image notes.
 *   2. generatePluginDSL() calls Qwen with golden Helix+Strata few-shots,
 *      returns a validated PluginDSL.
 *   3. <RenderDSL dsl={...}/> composes the React tree.
 *
 * Everything under this directory is self-contained — the existing
 * PluginCreator layout engine + svgComponentLibrary are untouched.
 * Wire this in behind a feature flag (see PluginCreator.js).
 */

export { RenderDSL, RenderDSLSafe } from './renderDSL';
export { validatePluginDSL, SCHEMA_FOR_PROMPT, MODULE_KINDS, ROW_KINDS } from './pluginDSL';
export { generatePluginDSL, warmup as warmupQwen } from './generatePluginDSL';
export { helixDSL } from './goldens/helix.dsl';
export { strataDSL } from './goldens/strata.dsl';
export {
  qwenVisionAnalyze,
  qwenVisionHealth,
  analyzePluginReferenceImage,
  fileToDataUrl,
} from '../../../../services/qwenChat';
