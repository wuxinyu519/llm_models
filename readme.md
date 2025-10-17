This repository organizes datasets used for **two-stage fine-tuning** and **final evaluation** of large language models (LLMs).  

Datasets are organized by pipeline stage: supervised fine-tuning (SFT), reinforcement learning with human feedback (RLHF/DPO), and final evaluation. See the directory layout below:

---

## 📦 Dataset Download

**Google Drive Link:** [Download Dataset](https://drive.google.com/file/d/1r6K_gRFq91jZENvZRF18ktchk-MI-s9F/view?usp=sharing)

Please download and extract the dataset before running the training scripts.

---

```
.
├── sft_with_miniplm/
│   └── after_miniplm
│       └──filtered.jsonl    # Stage 1: Supervised Fine-Tuning (SFT)
│
├── rhlf/
│   ├── data/                          
│       └── anotator/                       
│           └──dpo/                            # Stage 2: DPO / RLHF training data, gpt40-mini as judge to give reward from quality, diversity, and privacy.
│
└── eval/
    └── infinite_bench/
        └──infinitebench_gpt_gt                 # Stage 3: Evaluation dataset(s)
```

---

## 🚀 Stage 1 — Supervised Fine-Tuning (SFT)

**Script:** `./sft_with_miniplm/run_tune_gemma1b.sh`  
**Dataset:** `./sft_with_miniplm/after_miniplm_filtered.jsonl`

This stage performs **Supervised Fine-Tuning** (SFT) on high-quality instruction–response pairs filtered using the MiniPLM scoring mechanism.

**Usage:**
```bash
bash ./sft_with_miniplm/run_tune_gemma1b.sh
```

---

## 🚀 Stage 2 — Reinforcement Learning with Human Feedback (RLHF / DPO)

**Script:** `./rhlf/gemma1b_rhlf.sh`  
**Dataset:** `./rhlf/data/anotator/dpo/`

This stage performs alignment fine-tuning based on Direct Preference Optimization (DPO). It leverages human preference data to further align the model's behavior with human expectations after the SFT stage.

**Usage:**
```bash
bash ./rhlf/gemma1b_rhlf.sh
```

---

##  Stage 3 — Model Evaluation

**Script:** `./eval/infer.sh`  
**Dataset:** `./eval/infinite_bench/`

This stage performs model inference and accuracy evaluation on the InfiniteBench dataset, which measures the model's generalization, reasoning, and instruction-following capabilities.

**Usage:**
```bash
bash ./eval/infer.sh
```

---