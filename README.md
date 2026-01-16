# Synthetic Clinical Data Pipeline for Cognitive Risk Detection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)

## Project Overview

This repository contains a privacy-preserving pipeline designed to address data scarcity in medical Natural Language Processing (NLP). The project implements a synthetic data augmentation strategy to model linguistic biomarkers associated with cognitive impairment, specifically focusing on anomic aphasia and circumlocution.

By leveraging Large Language Models (LLMs) for data generation and Parameter-Efficient Fine-Tuning (PEFT), this framework allows for the development of robust detection models without requiring immediate access to sensitive, HIPAA-restricted patient data.

## Key Features

* **Synthetic Data Augmentation:** Implements a generation pipeline using GPT-4o with Chain-of-Thought (CoT) prompting to simulate pathological speech patterns, including specific disfluency markers (pauses, fillers) and syntactic simplification.
* **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes Low-Rank Adaptation (LoRA) to fine-tune foundation models (TinyLlama-1.1B) on synthetic clinical transcripts, significantly reducing trainable parameters while retaining model performance.

## Technical Architecture

The pipeline follows a Teacher-Student distillation approach:

1.  **Data Generation (Teacher):** A large foundation model (GPT-4o) generates high-fidelity transcripts based on standard clinical tasks (e.g., the "Cookie Theft" picture description). These transcripts are engineered to exhibit specific cognitive decline features.
2.  **Model Adaptation (Student):** A smaller, efficient base model is fine-tuned using the SFTTrainer from the Hugging Face TRL library.
3.  **Inference & Evaluation:** The adapted model is evaluated on its ability to generalize learned disfluency patterns to novel prompts.

## Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/synthetic-clinical-llm.git](https://github.com/yourusername/synthetic-clinical-llm.git)
cd synthetic-clinical-llm

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Generation
To generate the synthetic dataset using OpenAI's API:

```bash
python generate_data.py --num_samples 50 --output_file train.jsonl
```

### 2. Fine-Tuning
To run the LoRA fine-tuning pipeline:

```bash
python train.py --base_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --data_path "train.jsonl"
```

### 3. Inference
To test the model's output:

```bash
python inference.py --adapter_path "./results/my_mci_adapter" --prompt "Describe the cookie jar."
```

### Requirements
Python 3.8+
PyTorch 2.0+
Transformers
PEFT
TRL
OpenAI API Key (for data generation module only)
