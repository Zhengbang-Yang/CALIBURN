# CALIBURN
This repository provides training and evaluation scripts for applying **CALIBURN** to various datasets, including Harry Potter QA, raw Harry Potter text, and WMDP. It also includes evaluation guidance for unlearning performance.

---

## Datasets

- **Harry Potter QA**  
  Already included in the `data/` directory.

- **Harry Potter Text**  
  Download from: [https://github.com/swj0419/muse_bench](https://github.com/swj0419/muse_bench)

- **WMDP Dataset**  
  Download from: [https://github.com/centerforaisafety/wmdp](https://github.com/centerforaisafety/wmdp)

- **MUSE Testing Set**  
  Download from: [https://github.com/swj0419/muse_bench](https://github.com/swj0419/muse_bench)

- **Harry Potter Extend Testing Set**  
  Already included in the `data/` directory.

---

## CALIBURN Training

- For using CATNIP on the **Harry Potter QA** dataset, please refer to **scripts/HP_QA_CATNIP.sh**.
- For using CATNIP on the **Harry Potter text** dataset, please refer to **scripts/HP_raw_CATNIP.sh**.
- For using CATNIP on the **WMDP Cyber** dataset, please refer to **scripts/cyber_CATNIP.sh**.
- For using CATNIP on the **WMDP Bio** dataset (alternative setting), please refer to **scripts/bio_CATNIP.sh**.

---

## Evaluation

- To evaluate model **Know_f** on the **Harry Potter (MUSE testing set)**, please refer to **scripts/eval_knowmem_muse.sh**.
- To evaluate model **Know_f** on the **Harry Potter (Extend set)**, please refer to **scripts/eval_knowmem.sh**.
- To evaluate model performance on **WMDP** and **MMLU**, please refer to the official repo:  
  [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
