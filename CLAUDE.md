# CLAUDE.md

This file gives Claude Code (claude.ai/code) guidance for working in this repository.

## Project overview

`cognidist-id` classifies **cognitive distortions in Bahasa Indonesia text**. It is a 12-class sequence classification problem. The approach fine-tunes IndoBERT (`indobenchmark/indobert-base-p1`) and compares two `torchao` quantization approaches that use the same 4-bit weight-only config (`IntxWeightOnlyConfig`, int4, group size 32):

- **Quantization-aware training (QAT):** the model trains with simulated int4 weights, then is converted to int4.
- **Post-training quantization (PTQ):** a normal full-precision model is trained, then quantized to int4 without further training.

The full-precision model is also evaluated as a baseline.

The repo has only three files. There is no package, build system, test suite or `requirements.txt`:

- `cognitive-distortion-indonesia.ipynb`: the whole pipeline. It was first run on **Kaggle** (Tesla T4 GPU, Python 3.12, papermill) and has since been run on Colab (T4). All outputs are written under `OUT_PATH`, which is set in cell 2 (currently `/content`).
- `COGNITIVE DISTORTION DATASET IN BAHASA INDONESIA COMPLETE.csv`: the dataset (~1.1 MB, 4,992 rows).
- `README.md`: contains only the title.

## Dataset

Columns: `TEXT`, `DATA STATUS`, `FIRST ANNOTATOR`, `SECOND ANNOTATOR`.

- `TEXT`: free-text personal narratives in Indonesian. Every row has a span wrapped in `$...$` that marks the distorted thought. The notebook removes the `$` characters before tokenizing, so the span markers are not used as a signal at the moment.
- `DATA STATUS`: `RAW-ORI` (4,662 original rows) or `DIS-XX` (330 rows), where `XX` is a language code: ZH, EN, JV, TG, MS. These look like augmented variants, probably back-translated through that language. This has not been verified.
- The two annotator columns agree on 100% of rows. The notebook uses `FIRST ANNOTATOR` as the label.
- The classes are heavily imbalanced. `No Distortion` has 2,246 rows and `Fortune-telling` has 60. The other classes have 200–390 rows each.
- `LabelEncoder` assigns label IDs alphabetically: 0 All-or-nothing, 1 Discounting the positives, 2 Emotional Reasoning, 3 Fortune-telling, 4 Labeling, 5 Magnification or Minimization, 6 Mental filter, 7 Mind Reading, 8 No Distortion, 9 Overgeneralization, 10 Personalization and Blame, 11 Should statement.

## Notebook pipeline (cell order)

1. `pip install -U torchao`, then imports: pandas, torch, sklearn, `torchao.quantization` (`quantize_`, `IntxWeightOnlyConfig`, `PerGroup`, `QATConfig`).
2. Load the CSV from `./` (the relative path assumes the CSV sits next to the notebook). Label-encode `FIRST ANNOTATOR` into `LABEL` and build `label2id` / `id2label`.
3. Tokenizer: IndoBERT, `max_length=128`, truncation, padding.
4. Split: `train_test_split(test_size=0.1, random_state=42)`. It is **not stratified** and **not grouped**, so `DIS-*` augmented rows and their originals can end up in different splits. The split code is duplicated in two cells.
5. Model: `build_model()` creates a fresh IndoBERT classifier. `base_config` (int4, `PerGroup(32)`) is shared by QAT and PTQ so the comparison is fair. The QAT model is prepared with `quantize_(model, QATConfig(base_config, step="prepare"))`.
6. Training helper: `train(model, output_dir)` holds the shared HF `Trainer` setup and returns the trainer. Settings: lr 2e-5, batch size 16, 5 epochs, weight_decay 0.05, eval and save every epoch, best model chosen by `eval_accuracy`, `EarlyStoppingCallback(patience=2)`.
7. QAT run: train into `OUT_PATH/result-qat`, convert with `step="convert"`, then save with `save_model(model, OUT_PATH/qat-int4)`.
8. PTQ run: train a fresh `build_model()` into `OUT_PATH/result-fp32`. Save the unquantized model as `OUT_PATH/fp32` (the baseline). Then run `quantize_(ptq_model, base_config)` and save to `OUT_PATH/ptq-int4`.
9. Comparison: `get_model(path)` loads each of `fp32`, `ptq-int4` and `qat-int4` into a text-classification pipeline. Each one predicts on `test` (the 10% split) with `truncation=True, max_length=128`. The results go into the `comparison` DataFrame: accuracy, macro-F1, weighted-F1, `model.pt` size and test-set latency.
10. The last cell prints a classification report for each model and draws the three confusion matrices side by side.

## Known quirks and gotchas

- **Saving:** don't use `trainer.save_model` or `save_pretrained` on a quantized model. The torchao weights are wrapper tensor subclasses, and safetensors fails on them with `RuntimeError: Attempted to access the data pointer on an invalid python storage`. Use the notebook's `save_model` helper (`torch.save` of the state dict, plus config and tokenizer). Load with `get_model`, which uses `from_config` and `load_state_dict(assign=True)`.
- The two training runs need separate `output_dir`s. With `save_total_limit=1`, a shared directory would let one run delete the other's checkpoints.
- **Size and latency in the comparison:** `IntxWeightOnlyConfig` stores int4 values unpacked in int8, and the embeddings stay fp32, so `model.pt` shrinks by roughly half, not 8x. The int4 path dequantizes in plain PyTorch, so it can be slower than fp32 on a T4.
- **Don't switch back to `Int4WeightOnlyConfig`.** In torchao 0.18 its convert step needs the external `mslk` kernel package (`ImportError: Requires mslk >= 1.0.0`). It also needs bf16 weights and a newer GPU than Kaggle's T4. `IntxWeightOnlyConfig` is plain PyTorch.
- The epoch checkpoints in `result-qat/checkpoint-*` are saved **before** convert, so they hold the fake-quant (unconverted) model. Use `qat-int4` for the converted one.
- `compute_metrics` reports weighted precision, recall and F1. Given the class imbalance, macro-F1 is more informative when comparing runs.
- The commit `e90dd04` ("add quantization aware training") replaced most of the notebook's earlier contents. Check `git show 38cacdf:cognitive-distortion-indonesia.ipynb` for the earlier, non-QAT version, including any k-fold logic.

## Working with the notebook

- The notebook JSON is large, mainly because of saved outputs and widget state. To read or diff the code, extract the cell sources instead of opening the raw JSON:
  ```bash
  python -c "import json;[print(f'# [{i}]\n'+''.join(c['source'])) for i,c in enumerate(json.load(open('cognitive-distortion-indonesia.ipynb',encoding='utf-8'))['cells'])]"
  ```
- Edit cells with NotebookEdit and keep the cell order intact.
- Running locally needs a CUDA GPU plus `torch`, `torchao`, `transformers`, `datasets`, `scikit-learn`, `pandas` and `matplotlib`. Nothing is pinned. The original run used Kaggle docker image 31288.
- The notebook trains two models, roughly double the GPU time of a single run.
- Outputs under `OUT_PATH` (checkpoints and `model.pt` files) are not committed. Avoid committing model weights.
