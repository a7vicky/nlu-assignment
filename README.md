# Sanskrit-to-English Neural Machine Translation

## Assignment 2: Natural Language Understanding

NMT system for translating Sanskrit to English using a fine-tuned NLLB-200 pre-trained model.

## Architecture

### NLLB-200 Fine-tuned
- **Model**: `facebook/nllb-200-distilled-600M` (615M parameters)
- **Languages**: Sanskrit (`san_Deva`) to English (`eng_Latn`)
- **Fine-tuning**: 3 epochs, lr=2e-5, batch_size=8, weight_decay=0.01
- **Decoding**: Beam search with forced BOS token for English
- **Trainer**: HuggingFace `Seq2SeqTrainer` with `predict_with_generate`

## Setup

```bash
pip install -r requirements.txt
```

## Data

CSV files in the `data/` directory:
- `train_sa_10000.csv`, `train_en_10000.csv` (10,000 pairs)
- `dev_sa_1000.csv`, `dev_en_1000.csv` (1,000 pairs)
- `test_sa_1000.csv`, `test_en_1000.csv` (1,000 pairs)

Each CSV has columns: `Source_id` and `Sentence_sa`/`Sentence_en`.

## Usage

Run the Jupyter notebook `nllb_translation.ipynb` end-to-end. The notebook will:
1. Load the NLLB-200 model and tokenizer
2. Prepare training, dev, and test datasets
3. Fine-tune the model on the training data (3 epochs)
4. Generate translations and save `result/submission.csv`
5. Evaluate on the test set (BLEU, BERTScore)
6. Report efficiency metrics and show translation examples

## Output

- `result/submission.csv`: UTF-8 encoded file with `Source_id` and `Sentence_en` columns

## Pre-trained Models

- **`facebook/nllb-200-distilled-600M`** (615M parameters): Used as a pre-trained backbone for the translation model. Fine-tuned exclusively on the provided training dataset. No external data or APIs were used. The model natively supports Sanskrit and English.
- **`roberta-large`**: Used only by the `bert-score` evaluation library for computing BERTScore metrics. Not part of the translation pipeline.
