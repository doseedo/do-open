#!/bin/bash
# build.sh — build DoaeCodec.component and install to ~/Library/Audio/Plug-Ins/Components/
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
COMPONENT_NAME="DoaeCodec"
COMPONENT_BUNDLE="$BUILD_DIR/$COMPONENT_NAME.component"
INSTALL_DIR="$HOME/Library/Audio/Plug-Ins/Components"

ORT_INCLUDE="/opt/homebrew/include/onnxruntime"
ORT_LIB="/opt/homebrew/lib/libonnxruntime.dylib"
ONNX_MODEL="$HOME/Downloads/models_v2/onnx/oobleck_decoder_packed.onnx"

echo "=== Building DoaeCodec ==="

# --- Validate deps ---
[ -f "$ORT_LIB" ]    || { echo "ERROR: ONNX Runtime not found at $ORT_LIB"; exit 1; }
[ -d "$ORT_INCLUDE" ]|| { echo "ERROR: ONNX Runtime headers not found at $ORT_INCLUDE"; exit 1; }
[ -f "$ONNX_MODEL" ] || { echo "ERROR: Model not found at $ONNX_MODEL"; exit 1; }

# --- Create bundle structure ---
mkdir -p "$COMPONENT_BUNDLE/Contents/MacOS"
mkdir -p "$COMPONENT_BUNDLE/Contents/Resources"

# --- Compile ---
echo "Compiling..."
clang++ -std=c++17 \
    -arch arm64 \
    -target arm64-apple-macos14.0 \
    -dynamiclib \
    -install_name "@rpath/$COMPONENT_NAME.component/Contents/MacOS/$COMPONENT_NAME" \
    -fvisibility=hidden \
    -O2 \
    -I"$ORT_INCLUDE" \
    -I"$SCRIPT_DIR/Sources" \
    -framework CoreFoundation \
    -framework AudioToolbox \
    -framework Foundation \
    -L/opt/homebrew/lib \
    -lonnxruntime \
    "$SCRIPT_DIR/Sources/DoaeComponent.mm" \
    "$SCRIPT_DIR/Sources/DoaeAudioFile.mm" \
    "$SCRIPT_DIR/Sources/DoaeDecoder.mm" \
    -o "$COMPONENT_BUNDLE/Contents/MacOS/$COMPONENT_NAME"

echo "Compiled: $(du -sh "$COMPONENT_BUNDLE/Contents/MacOS/$COMPONENT_NAME" | cut -f1)"

# --- Copy resources ---
cp "$SCRIPT_DIR/Resources/Info.plist" "$COMPONENT_BUNDLE/Contents/Info.plist"
cp "$ONNX_MODEL" "$COMPONENT_BUNDLE/Contents/Resources/oobleck_decoder_packed.onnx"

echo "Model bundled: $(du -sh "$COMPONENT_BUNDLE/Contents/Resources/oobleck_decoder_packed.onnx" | cut -f1)"

# --- Embed ONNX Runtime (self-contained component) ---
# Resolve the real path behind the symlink to find the actual dylib name
# Resolve the recorded install name (may go through brew's opt symlink)
ORT_RECORDED="$(otool -L "$COMPONENT_BUNDLE/Contents/MacOS/$COMPONENT_NAME" | grep onnxruntime | awk '{print $1}')"
ORT_REAL="$(readlink -f "$ORT_LIB")"
ORT_DYLIB_NAME="$(basename "$ORT_REAL")"
cp "$ORT_REAL" "$COMPONENT_BUNDLE/Contents/MacOS/$ORT_DYLIB_NAME"
chmod u+w "$COMPONENT_BUNDLE/Contents/MacOS/$ORT_DYLIB_NAME"

# The linker recorded the real path; update it to @loader_path
install_name_tool \
    -change "$ORT_RECORDED" \
    "@loader_path/$ORT_DYLIB_NAME" \
    "$COMPONENT_BUNDLE/Contents/MacOS/$COMPONENT_NAME"

echo "ONNX Runtime embedded: $(du -sh "$COMPONENT_BUNDLE/Contents/MacOS/$ORT_DYLIB_NAME" | cut -f1)"

# --- Install ---
echo ""
echo "=== Installing to $INSTALL_DIR ==="
mkdir -p "$INSTALL_DIR"
rm -rf "$INSTALL_DIR/$COMPONENT_NAME.component"
cp -R "$COMPONENT_BUNDLE" "$INSTALL_DIR/"

echo ""
echo "=== Done ==="
echo "Component installed: $INSTALL_DIR/$COMPONENT_NAME.component"
echo ""
echo "To test:"
echo "  python3 doae/test_create.py          # create test .doae"
echo "  afplay /tmp/doae_test/test_session.doae   # play via Core Audio"
echo ""
echo "To uninstall:"
echo "  rm -rf '$INSTALL_DIR/$COMPONENT_NAME.component'"
