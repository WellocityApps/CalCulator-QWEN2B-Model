#!/usr/bin/env bash
# Runs the host CLI over a CSV of Nutrition5k dishes for each model variant.
# usage: tools/compare_quant.sh <cli> <mmproj> <images_dir> <have.csv> <out.jsonl> <model.gguf>...
set -euo pipefail
CLI=$1; MMPROJ=$2; DIR=$3; CSV=$4; OUT=$5; shift 5
: > "$OUT"
for MODEL in "$@"; do
  while IFS=, read -r id mass cal carb prot fat ing; do
    for variant in 0 1; do
      note=""
      [ "$variant" = 1 ] && note="Meal description: $(echo "$ing" | sed 's/;/; /g')"
      line=$("$CLI" "$MODEL" "$MMPROJ" "$DIR/$id.png" "$note" 2>/dev/null | tail -1)
      echo "{\"model\":\"$(basename "$MODEL")\",\"id\":\"$id\",\"variant\":$variant,\"true\":[$mass,$cal,$carb,$prot,$fat],\"run\":$line}" >> "$OUT"
    done
  done < "$CSV"
done
