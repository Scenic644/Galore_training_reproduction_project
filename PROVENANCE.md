# Code Provenance and Attribution

### 1. External Packages and Upstream Implementations
* **GaLore Algorithm (`galore_torch`):**
  * Source: Official GaLore Repository (Zhao et al., 2024, ICML)
  * Package: `galore-torch==1.0`
  * Role: Provides the low-rank gradient projection operator and GaLoreAdamW optimizer. Reused as-is via pip.
* **LoRA Implementation (`peft`):**
  * Source: Hugging Face PEFT Library (Hu et al., 2022)
  * Package: `peft>=0.8.0`
  * Role: Injects trainable low-rank adaptation matrices $A$ and $B$ into RoBERTa attention layers. Reused as-is.
* **Base Model & Data Pipeline:**
  * Source: Hugging Face `transformers` and `datasets`
  * Model: `roberta-base`
  * Dataset: `nyu-mll/glue` (SST-2 task)

### 2. Team-Authored Code
* `train.py`: Authored by our team. Implements the training loop, memory tracking (`torch.cuda.max_memory_allocated()`), evaluation pipeline, and results serialization.
* `sanity_check.py`: Authored by our team. Performs minimal forward/backward pass verification to detect shape mismatches or memory leaks prior to training.
* `galore_reproduction_colab.ipynb`: Authored by our team. Coordinates execution on Google Colab T4 GPU.
