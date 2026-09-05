# AiSOME Neural Nexus

Multilingual **climate-change stance detection** on social media comments in **English, Hindi, and Bangla**, using domain-adapted transformer language models and a weighted ensemble.

Each comment is classified into one of three stance labels:

| Label | Meaning |
|---|---|
| `FAVOR` | Comment supports / agrees with climate action or climate science |
| `AGAINST` | Comment opposes / denies climate action or climate science |
| `NONE` | Comment is neutral or unrelated to the climate stance question |

## Overview

The project fine-tunes three multilingual transformer encoders for stance classification, then combines their predictions with a **soft-voting ensemble** to get a more robust final label. Each model goes through two stages:

1. **Domain-adaptive pretraining (MLM)** — continue pretraining the base checkpoint with masked-language-modeling on a combined corpus of English, Hindi, and Bangla climate-related text (including machine-translated English→Hindi / English→Bangla data) so the model picks up domain and script-specific vocabulary.
2. **Fine-tuning** — train a 3-class sequence-classification head (`AGAINST` / `FAVOR` / `NONE`) on labeled data, evaluated with stratified cross-validation.

The three fine-tuned models are then combined in an ensemble notebook that runs **soft voting** and **performance-weighted soft voting**, and can also score new, unlabeled Hindi/Bangla files.

## Models

| Model | Base checkpoint | Notes |
|---|---|---|
| XLM-R | `xlm-roberta-base` | Baseline multilingual encoder |
| IndicBERT | `ai4bharat/IndicBERTv2-MLM-only` | Indic-language-focused encoder |
| MuRIL | `google/muril-base-cased` | Google's multilingual Indic-language encoder |

Ensemble weights are derived from each model's validation performance:

```
Model1 (IndicBERT): 0.885
Model3 (MuRIL):      0.888
Model5 (XLM-R):       0.854
```

## Repository Structure

```
AISOME/
├── model-1/                 # XLM-R — initial pretraining + single-split fine-tuning
│   ├── xlmr-Pretraining.ipynb
│   ├── xlmr-Finetuning (without pre-process+weighted fine tune).ipynb
│   ├── Hindi_xlmr.xlsx           # predictions on Hindi test comments
│   └── Bangla_xlmr.xlsx          # predictions on Bangla test comments
│
├── model-2/                 # IndicBERT — pretraining + cross-validated fine-tuning
│   ├── indicBert-Pretraining.ipynb
│   ├── indicBert-Finetuning (CV).ipynb
│   ├── Hindi_indicBert.xlsx
│   └── Bangla_indicBert.xlsx
│
└── model-3/                 # Cross-validated versions of all 3 models + ensemble
    ├── xlmr_cv/
    │   ├── xlmr_Pretraining.ipynb
    │   └── xmlr_Finetuning (CV).ipynb
    ├── indicBert_cv/
    │   ├── indicBert_Pretraining.ipynb
    │   └── indicBert_Finetuning (CV).ipynb
    ├── muril_cv/
    │   ├── muril_Pretraining.ipynb
    │   └── muril_Finetuning (CV).ipynb
    ├── ENSEMBLE (CV).ipynb           # soft-voting / weighted soft-voting ensemble
    ├── Hindi_Ensemble (CV).csv       # final ensembled predictions (Hindi)
    └── Bangla_Ensemble (CV).csv      # final ensembled predictions (Bangla)
```

## Pipeline

```
English / Hindi / Bangla         ┌────────────────────┐
climate text corpus  ───────────▶│  MLM Pretraining    │  (per model)
(incl. translated data)          └─────────┬──────────┘
                                            ▼
                                  ┌────────────────────┐
                       Labeled ──▶│  Fine-tuning        │
                       stance     │  (Stratified CV)    │  (per model)
                       dataset    └─────────┬──────────┘
                                            ▼
                        XLM-R  ─┐  ┌────────────────────┐
                        IndicBERT│▶│  Soft-voting /       │
                        MuRIL   ─┘  │  Weighted ensemble  │
                                    └─────────┬──────────┘
                                              ▼
                                   Final stance label per
                                   comment (+ confidence,
                                   agreement metrics)
```

The `ENSEMBLE (CV).ipynb` notebook also:
- Normalizes label variants (e.g. `FAVOUR` → `FAVOR`) and cleans text (strips URLs, zero-width characters, etc.)
- Builds/reuses cached stratified evaluation folds with a dataset fingerprint, so folds aren't silently reused after the data changes
- Uses dynamic padding, mixed precision, and automatic batch-size fallback for GPU memory efficiency
- Produces per-model and per-ensemble metrics (accuracy, precision, recall, F1), classification reports, and agreement/confidence plots
- Scores new unlabeled Hindi/Bangla CSV or Excel files with auto-detected text columns

## Getting Started

These notebooks were built for **Google Colab** (they mount Google Drive for storage) but also run locally/on Jupyter if the expected files exist relative to the working directory.

### Requirements

```bash
pip install "transformers>=4.48,<5" "accelerate>=1.2,<2" \
    datasets sentencepiece evaluate scikit-learn openpyxl xlrd joblib \
    torch pandas numpy matplotlib
```

(Each notebook also has its own `%pip install` / `!pip install` cell at the top — run that cell first.)

### Data expected

The notebooks expect the following input files (not included in this repo — supply your own):

- `English_train_data.csv` — English climate comments (`sentence`, `label`)
- `final_dataset_hindi1.csv` / `english_to_hindi.csv` — native + translated Hindi text
- `merged_bengali_dataset.csv` / `english_to_bengali.csv` — native + translated Bangla text
- `non_climate_none.csv` — off-topic examples for the `NONE` class
- `Hindi_500_test data.xlsx`, `Bangla_500_test data.xlsx` — unlabeled comments to predict on

### Running

1. **Pretrain** — open a model's `*-Pretraining.ipynb`, mount Drive/set paths, run all cells to continue MLM pretraining on the combined corpus.
2. **Fine-tune** — open the matching `*-Finetuning*.ipynb`, point it at the pretrained checkpoint, and run cross-validated fine-tuning for stance classification.
3. **Ensemble** — open `model-3/ENSEMBLE (CV).ipynb`, set `MODEL_PATHS` to your three fine-tuned checkpoint folders, then run all cells to fold-predict, ensemble, evaluate, and score new files.

## Outputs

- `predictions/fold_*` — per-fold, per-model probability + prediction caches
- `reports/` — metrics CSVs, classification reports, agreement/confidence plots
- `test_predictions/` — predictions on new Hindi/Bangla files
- `Hindi_Ensemble (CV).csv`, `Bangla_Ensemble (CV).csv` — final per-comment predictions from every model plus the ensemble, with per-class probabilities, vote counts, and agreement type (e.g. `3/3 Unanimous`, `2/3 Majority`)

## Tech Stack

- Python, PyTorch, Hugging Face `transformers` / `datasets` / `accelerate` / `evaluate`
- scikit-learn (stratified k-fold, metrics)
- pandas / NumPy / openpyxl / xlrd for data handling
- matplotlib for evaluation plots
- Google Colab + Google Drive for training/storage

## Notes

- No `requirements.txt` or `LICENSE` file is currently included in this repo — consider adding both.
- File paths (e.g. `/content/drive/MyDrive/...`) are Colab-specific; update `ROOT` and the data-path candidates in each notebook if running elsewhere.
