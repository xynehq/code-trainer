# CPT-LoRA Fine-Tuning Scripts

This repository provides scripts and resources for fine-tuning CPT (Continued Pre-training) models using LoRA (Low-Rank Adaptation) methods, focusing on flexibility, efficiency, and reproducibility.

## Repository Structure

The repository contains two main folders, each dedicated to a specific fine-tuning scenario:

### 1. [`CPT-KwaiPilot`](https://github.com/AdityaNarayan001/CPT-LoRA_FineTuning_Scripts/tree/main/CPT-KwaiPilot)
This folder includes scripts and resources designed for fine-tuning CPT models in the context of the KwaiPilot adaptation or dataset. It may contain code for data handling, training, and evaluation tailored to this specific use case.

### 2. [`CPT_Qwen-2.5-Coder-Instruct-32B`](https://github.com/AdityaNarayan001/CPT-LoRA_FineTuning_Scripts/tree/main/CPT_Qwen-2.5-Coder-Instruct-32B)
This folder focuses on fine-tuning CPT models using the Qwen-2.5-Coder-Instruct-32B configuration or dataset. The scripts here are likely specialized for instruction-based or coder-centric training workflows.

### 3. [`EulerDatasetGen`](./EulerDatasetGen)
This folder contains tools for downloading Haskell packages from Hackage and preparing them into training datasets for Large Language Models (LLMs). It provides automated package downloading, source code processing, and dataset preparation optimized for Continued Pre-Training (CPT) tasks with memory-efficient streaming processing.

---

Each folder contains its own scripts, configurations, and documentation for running experiments and training models.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xynehq/code-trainer.git
   cd code-trainer
   ```

2. **Install dependencies:**  
   Refer to the individual README files in each folder for environment setup and installation instructions.

3. **Run a script:**  
   Scripts can be executed from their respective folders. Please consult the folder-specific documentation for details.

## Contribution

Contributions, suggestions, and feature requests are welcome! Feel free to open issues or submit pull requests to improve this repository.
