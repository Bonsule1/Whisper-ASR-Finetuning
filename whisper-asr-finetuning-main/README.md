# Whisper ASR Fine-Tuning

Fine-tuning OpenAI's Whisper (small) model for Automatic Speech Recognition on the LibriSpeech dataset, built with PyTorch and Hugging Face Transformers.

## Overview

This project covers the full ASR pipeline:
- Loading and exploring LibriSpeech (`train-clean-100`, dev, and test splits)
- Audio preprocessing and feature extraction (log-Mel spectrograms via Whisper's feature extractor)
- Text normalization (abbreviation expansion, casing, punctuation)
- A custom data collator for padding audio features and label sequences
- Fine-tuning Whisper with Hugging Face's `Seq2SeqTrainer`
- Evaluation using Word Error Rate (WER) and Character Error Rate (CER)

## Tech Stack

- Python, PyTorch
- Hugging Face Transformers, Datasets, Evaluate
- Librosa (audio/signal processing)
- jiwer (WER/CER computation)

## Dataset

[LibriSpeech ASR](https://huggingface.co/datasets/openslr/librispeech_asr) — `train.clean.100` for training, `validation.clean` and `test.clean` for evaluation, streamed directly from Hugging Face Datasets.

## Approach

The model is fine-tuned starting from `openai/whisper-small`. Audio is capped at 30 seconds and converted to log-Mel features; transcripts are normalized (lowercased, common abbreviations expanded) before tokenization. Training uses mixed precision (fp16), gradient accumulation, and early stopping based on validation WER.

## Results

Final performance is reported as WER and CER on the held-out test split (see the notebook for exact numbers and example predictions).

## Notes

This was built as part of a Speech Processing course, covering the theory behind classical ASR (HMM/GMM acoustic modeling, MFCCs, language modeling) alongside this hands-on deep learning implementation using modern Transformer-based architectures.
