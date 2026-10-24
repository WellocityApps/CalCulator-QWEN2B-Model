# CalCulator Nutrition Model (Qwen3.5-2B)

A fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) that estimates the nutrition of a meal from a photo. A short text note is optional. The model returns a single JSON object:

```json
{"total_mass_g":272.0,"calories_kcal":298.8,"carbs_g":13.7,"protein_g":39.1,"fat_g":10.8}
```

It runs on-device in the CalCulator app through llama.cpp, using a Q4_K_M text model and an F16 vision projector (about 1.9 GB in total).

## Downloads

The weights are attached to the [GitHub Release](https://github.com/WellocityApps/CalCulator-QWEN2B-Model/releases/tag/v2026.10.24-q4km) `v2026.10.24-q4km`, because git cannot hold files of this size.

| File | Size | Use |
|---|---|---|
| `calc-nutrition-qwen35-2b-q4_k_m.gguf` | 1.27 GB | Text model, Q4_K_M (llama.cpp) |
| `calc-nutrition-qwen35-2b-mmproj-f16.gguf` | 668 MB | Vision projector, F16 (llama.cpp `--mmproj`) |
| `model-0000{1,2,3}-of-00003.safetensors` | 4.43 GB | Full BF16 weights (Transformers). Put them next to the configs in [`model/bf16/`](model/bf16/) |

Each file can be downloaded directly:

```
https://github.com/WellocityApps/CalCulator-QWEN2B-Model/releases/download/v2026.10.24-q4km/<file>
```

Checksums are listed in [`SHA256SUMS`](SHA256SUMS).

## Results

The test set has **1,283 held-out photos**: 992 from Nutrition5k and 291 from SNAPMe. It uses a grouped split, so no dish group appears in both train and test. Inference used BF16 weights with thinking disabled. Thinking was turned off because it scored worse on validation.

Two input variants were tested:

- **Photo only.** The model gets the image and nothing else.
- **Photo + description.** The model also gets the dish's ingredient list as a note, e.g. `Meal description: chicken; eggplant; olive oil; …`.

The metrics are:

- **MAE**: mean absolute error.
- **WAPE**: total absolute error ÷ total true value.

Lower is better for both.

### All test photos

| Target | MAE (photo only) | WAPE (photo only) | MAE (photo + description) | WAPE (photo + description) |
|---|---:|---:|---:|---:|
| Calories | 69.3 kcal | 31.0 % | **63.0 kcal** | **28.4 %** |
| Mass | 55.5 g | 28.0 % | 55.7 g | 28.1 % |
| Carbs | 7.4 g | 37.2 % | **7.1 g** | **35.7 %** |
| Protein | 4.8 g | 34.2 % | **4.5 g** | **32.5 %** |
| Fat | 4.3 g | 42.2 % | **3.9 g** | **38.2 %** |
| Valid JSON | 99.2 % (10 failures) | | 99.7 % (4 failures) | |

### By dataset (calories)

| Dataset | n | Photo only | Photo + description |
|---|---:|---|---|
| Nutrition5k (cafeteria plates, overhead camera) | 992 | 53.9 kcal MAE · 26.8 % WAPE | **48.6 kcal MAE · 24.4 % WAPE** |
| SNAPMe (real-world phone photos) | 291 | 121.3 kcal MAE · 40.5 % WAPE | **112.2 kcal MAE · 37.5 % WAPE** |

The full per-dataset metrics for every target, including RMSE, are in [`model/bf16/nutrition_spec.json`](model/bf16/nutrition_spec.json).

The mass RMSE is large (about 329 g) even though the MAE is about 55 g. That gap comes from a few Nutrition5k outliers, which are very heavy plates.

### Quantization check (Q4_K_M vs BF16)

Both GGUF builds were run with llama.cpp on 12 Nutrition5k dishes, with each input variant, on an Intel Core i7-9750H laptop CPU ([`validation/host_bf16_vs_q4km.jsonl`](validation/host_bf16_vs_q4km.jsonl)).

| | BF16 GGUF | Q4_K_M GGUF |
|---|---:|---:|
| Calories WAPE (photo only / photo + description) | 16.5 % / 14.5 % | 19.4 % / 17.6 % |
| Mass WAPE (photo only / photo + description) | 16.9 % / 15.4 % | 14.9 % / 18.0 % |
| Median prefill (image + prompt) | 17.2 s | 11.1 s |
| Median decode (~47 tokens) | 10.8 s | 3.8 s |
| Peak RAM | 4.5 GB | 2.7 GB |
| Text-model file size | 3.78 GB | 1.27 GB |

On this small sample, quantization costs about 3 points of calorie WAPE. In return the model uses 40 % less RAM and decodes about 3× faster.

Text-only use, with no image, also works: [`validation/host_text_only_probes.jsonl`](validation/host_text_only_probes.jsonl). For example, `1 banana` → 117 g, 80 kcal, 21.6 g carbs.

## Training

The full notebook is in [`notebook/Qwen_Nutrition_A100_EN.ipynb`](notebook/Qwen_Nutrition_A100_EN.ipynb), with its outputs kept. The Python modules it embeds are also included as plain files in [`training/`](training/).

**Setup**

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-2B` @ `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Hardware | 1 × NVIDIA A100 (Colab) |
| Data | 6,415 images: Nutrition5k (4,959) and SNAPMe (1,459, loss weight 0.30). MetaFood3D was not used. |
| Split | Grouped 72 / 8 / 20: 4,619 train, 513 validation, 1,283 test |
| Trainable | Last 12 text layers, last 12 vision blocks, final norm, and vision merger |
| Batch | Effective 16 (1 × 16 accumulation on 40 GB, or 2 × 8 on 80 GB). Max length 1,536 tokens. Seed 42. |
| Image size | 128² to 512² pixels (processor `min_pixels` / `max_pixels`) |

**Curriculum**

Training ran in stages. An auxiliary MLP regression head teaches the backbone the numeric targets first. The head is then removed, and the model is fine-tuned to produce JSON only.

| Stage | Epochs | LR (backbone / head) | Loss |
|---|---:|---|---|
| `02_head` | 3 | 0 / 1e-3 | Regression |
| `03_frozen_head` | 1 | 3e-6 / 0 | Regression |
| `04_joint_bridge` | 2 | 5e-6 / 1e-4 | 0.5 × regression + JSON cross-entropy |
| `05_json` | 1 | 2e-6 / (head removed) | JSON cross-entropy |

The prompt notes during training were mixed:

- 50 % no note
- 30 % an English ingredient list
- 20 % a verified preparation note or a Hebrew ingredient list

Also, 35 % of SFT targets came with a short reasoning trace that breaks the meal into components.

### Prompt format

To get results close to the ones above, use the same prompt as training:

- **System:**
  > Estimate the edible meal in the image, using user notes when supplied. Preparation notes override visual guesses: respect explicit statements such as no added oil. If no note is supplied, estimate from the image. Return exactly one final JSON object with keys total_mass_g, calories_kcal, carbs_g, protein_g, fat_g. All values are finite nonnegative numbers. Grams for mass/macros, kcal for energy. No markdown; no explanations inside the final JSON.
- **User:** `<image>` + `Estimate the meal.\n` + an optional note, such as `Meal description: …`.
- **Chat template:** use the one shipped in the GGUF or in [`model/bf16/chat_template.jinja`](model/bf16/chat_template.jinja), with `enable_thinking=False`.

### Reproducing the GGUF files

[`scripts/convert_model.sh`](scripts/convert_model.sh) converts the BF16 export with llama.cpp's `convert_hf_to_gguf.py` and then runs `llama-quantize … Q4_K_M`. The script was written for the app repository, so adjust the paths before running it.

## Limitations

- **No before/after comparison.** The comparison against the original Qwen3.5-2B on the same test set was started but did not finish (1,178 of 1,283 photos), so no improvement over the base model is reported here.
- **SNAPMe descriptions can be too informative.** Some SNAPMe descriptions say more than a real user would type. That can make the "photo + description" numbers look better than real use.
- **JSON is validated, not enforced.** It is checked after generation, not guaranteed by constrained decoding. About 0.3–0.8 % of outputs on the test set were not valid JSON.
- **Estimates only.** The outputs are not medical or dietary advice.

## License and credit

Everything in this repository is released under the [Apache License 2.0](LICENSE): code, notebook, documentation, and the model weights published in the release. You can use, modify, redistribute, and sell it, including in closed-source and commercial products.

**Credit is required.** If you redistribute this work or anything derived from it (for example fine-tuned or re-quantized weights), you must include the [`NOTICE`](NOTICE) file. It credits this repository, the Qwen3.5-2B base model, and the training datasets.

| Component | License | Credit |
|---|---|---|
| Base model: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache 2.0 | Alibaba Cloud (Qwen team) |
| Dataset: [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) | CC BY 4.0 | Thames et al., CVPR 2021 |
| Dataset: [SNAPMe](https://doi.org/10.15482/USDA.ADC/1528346) | CC BY-SA 4.0 | Chin, Larke et al., Ag Data Commons 2022; Larke et al., *Nutrients* 2023 |

This repository contains no dataset images. If you retrain with the datasets themselves, follow their licenses. In particular, SNAPMe is ShareAlike: material adapted from it must be shared under CC BY-SA 4.0.

### Citation

```bibtex
@misc{calculator_qwen2b_nutrition_2026,
  title  = {CalCulator Nutrition Model: Qwen3.5-2B fine-tuned for meal nutrition estimation},
  author = {WellocityApps},
  year   = {2026},
  url    = {https://github.com/WellocityApps/CalCulator-QWEN2B-Model}
}
```

Please also cite [Nutrition5k](https://arxiv.org/abs/2103.03375) and [SNAPMe](https://www.mdpi.com/2072-6643/15/23/4972).
