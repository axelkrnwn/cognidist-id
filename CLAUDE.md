# CLAUDE.md

This file gives Claude Code (claude.ai/code) guidance for working in this repository.

## Project overview

`cognidist-id` classifies **cognitive distortions in Bahasa Indonesia text**. It is a 12-class sequence classification problem. The approach fine-tunes IndoBERT (`indobenchmark/indobert-base-p1`) with **quantization-aware training (QAT)** from `torchao`, using a 4-bit weight-only config (`IntxWeightOnlyConfig`, int4, group size 32).

The repo has only three files. There is no package, build system, test suite or `requirements.txt`:

- `cognitive-distortion-indonesia.ipynb`: the whole pipeline. It was written for and run on **Kaggle**: Tesla T4 GPU, Python 3.12, papermill, with the dataset attached as a Kaggle data source.
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
5. Model: `AutoModelForSequenceClassification`. QAT prepare step is `quantize_(model, QATConfig(IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32)), step="prepare"))`.
6. Train with HF `Trainer` using lr 2e-5, batch size 16, 5 epochs, weight_decay 0.05, eval and save every epoch, best model chosen by `eval_accuracy`, `EarlyStoppingCallback(patience=2)`, and output written to `./result`. After training, the QAT convert step runs (`quantize_(..., step="convert")`), and the converted model is saved to `./result/qat-int4`.
7. Inference and evaluation: `get_model('./result/qat-int4')` builds a text-classification pipeline. It predicts on `test` (the 10% split) with `truncation=True, max_length=128`. Predicted label names are mapped back to IDs through `label2id` to get `final_preds`. The notebook then prints accuracy, a classification report and a confusion matrix.

## Known quirks and gotchas

- The evaluation cell's print still says "Ensemble Accuracy". That wording is left over from an earlier k-fold or ensemble version; it is a single model now.
- It's unverified whether `from_pretrained` restores torchao's quantized tensor subclasses from `./result/qat-int4`. If the reloaded model's metrics don't match `val_metrics` from cell 15, run the pipeline on the in-memory `model` instead.
- **Don't switch back to `Int4WeightOnlyConfig`.** In torchao 0.18 its convert step needs the external `mslk` kernel package (`ImportError: Requires mslk >= 1.0.0`). It also needs bf16 weights and a newer GPU than Kaggle's T4. `IntxWeightOnlyConfig` is plain PyTorch.
- The epoch checkpoints in `./result/checkpoint-*` are saved **before** convert, so they hold the fake-quant (unconverted) model. The converted model is the in-memory `model`, or `./result/qat-int4`.
- `compute_metrics` reports weighted precision, recall and F1. Given the class imbalance, macro-F1 is more informative when comparing runs.
- The commit `e90dd04` ("add quantization aware training") replaced most of the notebook's earlier contents. Check `git show 38cacdf:cognitive-distortion-indonesia.ipynb` for the earlier, non-QAT version, including any k-fold logic.

## Working with the notebook

- The notebook JSON is large, mainly because of saved outputs and widget state. To read or diff the code, extract the cell sources instead of opening the raw JSON:
  ```bash
  python -c "import json;[print(f'# [{i}]\n'+''.join(c['source'])) for i,c in enumerate(json.load(open('cognitive-distortion-indonesia.ipynb',encoding='utf-8'))['cells'])]"
  ```
- Edit cells with NotebookEdit and keep the cell order intact.
- Running locally needs a CUDA GPU plus `torch`, `torchao`, `transformers`, `datasets`, `scikit-learn`, `pandas` and `matplotlib`. Nothing is pinned. The original run used Kaggle docker image 31288.
- `./result/` (checkpoints) is not committed. Avoid committing model weights.
