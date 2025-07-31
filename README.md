# SurgFed: Language-guided Multi-Task Federated Learning for Surgical Video Understanding

<p align="center">
    <img src="https://img.shields.io/github/license/yourname/SurgFed?style=flat" alt="License"/>
</p>

**SurgFed** is a multi-task federated learning framework designed for surgical video understanding. It incorporates language-guided semantic priors to collaboratively perform multiple tasks (e.g., surgical scene segmentation and depth estimation) across decentralized clinical datasets while preserving data privacy. SurgFed demonstrates strong generalization and cross-institutional robustness on various real-world surgical video datasets.

---

## 🥉 Environment Requirements

Please refer to the [Medical-SAM2](https://github.com/SuperMedIntel/Medical-SAM2) repository for detailed environment setup instructions.

---

## 🚀 How to Run

Run the training with:

```bash
CUDA_VISIBLE_DEVICES=0 python train_hnfl.py \
    -net sam2 \
    -exp_name /exp_name \
    -sam_ckpt ./checkpoints/sam2_hiera_small.pt \
    -sam_config sam2_hiera_s_dep \
    -image_size 512 \
    -val_freq 1 \
    -prompt bbox \
    -prompt_freq 2 \
    -Layers both \
    -num_nets 5 \
    -local_epochs 3
```

---


