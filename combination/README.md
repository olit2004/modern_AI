# Verified Recursive Self-Improvement: Amharic Qwen Student

Combines verified recursive self-learning with low-resource Amharic distillation for `Qwen/Qwen2.5-0.5B-Instruct`.

## Contents

- `recursive_amharic_qwen_colab.ipynb`: Google Colab notebook with GPU support and automatic results download.
- `data/training_data.csv`: 1,843 clean teacher demonstration pairs.
- `data/evaluation_prompts.csv`: 60 held-out native Amharic evaluation prompts.
- `verifier.py`: Linguistic referee module (Ethiopic Unicode purity, loop detection, alignment).

## Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com) and upload `combination/recursive_amharic_qwen_colab.ipynb`.
2. Set runtime accelerator to T4 GPU (`Runtime` > `Change runtime type` > `T4 GPU`).
3. Click `Runtime` > `Run all`. The notebook automatically downloads the dataset if not present.
4. Total runtime is approximately 15 minutes on T4 GPU.
5. When complete, `amharic_recursive_results.zip` is automatically downloaded containing the final adapter, evaluation predictions, and experiment summary.
