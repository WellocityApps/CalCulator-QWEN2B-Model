#!/usr/bin/env bash
# Converts the fine-tuned HF export (BF16 safetensors) into the mobile GGUF
# files the app loads, and prints their sizes and SHA-256 (paste into
# lib/src/model_install/model_manifest.dart when the model changes).
#
# usage: tools/convert_model.sh [export_dir] [out_dir]
#   export_dir defaults to develop_resources/export_bf16
#   out_dir    defaults to models/dist/<version>
#
# Requirements: python3.12 (uv recommended), cmake + ninja (Android SDK's cmake
# works), ~10 GB free disk. On Intel macOS the last PyTorch wheel is 2.2.2,
# which needs numpy<2; this script installs that combination.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/develop_resources/export_bf16}"
VERSION="2026.10.24-q4km"
OUT="${2:-$ROOT/models/dist/$VERSION}"
WORK="$ROOT/models/work"
LLAMA="$ROOT/third_party/llama.cpp"
mkdir -p "$OUT" "$WORK"

"$ROOT/tools/fetch_llama_cpp.sh"

# 1. Python env for llama.cpp's converter.
VENV="$WORK/convenv"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -p 3.12 "$VENV"
  if [ "$(uname -s)-$(uname -m)" = "Darwin-x86_64" ]; then
    VIRTUAL_ENV="$VENV" uv pip install "torch==2.2.2" "numpy==1.26.4" "sentencepiece<0.3" \
      "transformers==4.57.6" "protobuf<5" safetensors
  else
    VIRTUAL_ENV="$VENV" uv pip install -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"
  fi
fi

# 2. The export was saved with transformers 5.x, whose tokenizer_config names
#    the class "TokenizersBackend". The converter's transformers 4.57 only
#    needs to read tokenizer.json, so expose the export through a shadow
#    directory whose tokenizer_config.json says Qwen2TokenizerFast. Weights and
#    tokenizer.json are symlinked unchanged.
SHADOW="$WORK/export_shadow"
rm -rf "$SHADOW"; mkdir -p "$SHADOW"
for f in "$SRC"/*; do ln -s "$f" "$SHADOW/"; done
rm "$SHADOW/tokenizer_config.json"
python3 -I - "$SRC/tokenizer_config.json" "$SHADOW/tokenizer_config.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); d["tokenizer_class"] = "Qwen2TokenizerFast"; d.pop("backend", None)
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY

# 3. Convert: text model BF16 (lossless), vision projector F16 (as planned in
#    the training notebook's deploy.py).
"$VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$SHADOW" --outfile "$WORK/text-bf16.gguf" --outtype bf16
"$VENV/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$SRC" --mmproj --outfile "$WORK/mmproj-f16.gguf" --outtype f16

# 4. Quantize text weights to Q4_K_M.
CMAKE="$(command -v cmake || ls "$HOME"/Library/Android/sdk/cmake/*/bin/cmake | tail -1)"
"$CMAKE" -S "$LLAMA" -B "$LLAMA/build-host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_OPENSSL=OFF
"$CMAKE" --build "$LLAMA/build-host" --target llama-quantize -j 8
"$LLAMA/build-host/bin/llama-quantize" "$WORK/text-bf16.gguf" "$WORK/text-q4_k_m.gguf" Q4_K_M

cp "$WORK/text-q4_k_m.gguf" "$OUT/calc-nutrition-qwen35-2b-q4_k_m.gguf"
cp "$WORK/mmproj-f16.gguf" "$OUT/calc-nutrition-qwen35-2b-mmproj-f16.gguf"
echo "== files for model_manifest.dart"
for f in "$OUT"/*.gguf; do
  printf '%s bytes=%s sha256=%s\n' "$(basename "$f")" "$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f")" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
done
