\# Chest X-ray Dataset Preprocessing and 1-GPU Benchmark



\## Overview



This folder contains the preprocessing scripts, dataset loader, model definition, and single-GPU benchmark used to prepare the NIH Chest X-ray dataset for our distributed deep learning experiments.



The goal of Phase 1 was to clean and organize the dataset, preprocess the images, and establish a single-GPU training throughput baseline for later multi-GPU comparisons.



\## 1. Dataset Preparation



The NIH Chest X-ray dataset contains \*\*112,120 chest X-ray images\*\* with 14 disease labels.



The following scripts were used:



| File | Description |

|---|---|

| `01\_clean\_metadata.py` | Cleans and prepares the original dataset metadata. |

| `02\_split\_dataset.py` | Creates the training, validation, and test splits. |

| `03\_label\_frequencies.py` | Calculates disease label frequencies for analyzing class imbalance. |

| `04\_resize\_images.py` | Resizes and pads images to 224 × 224 pixels. |



The processed metadata and dataset splits are saved as:



\- `cleaned\_metadata.csv`

\- `train.csv`

\- `val.csv`

\- `test.csv`

\- `label\_frequencies.csv`



The training split contains \*\*77,988 images\*\*.



All \*\*112,120 images\*\* were successfully preprocessed into 224 × 224 grayscale images while preserving their aspect ratios through padding.



\## 2. Dataset Loader and Model



The `dataset.py` script loads the preprocessed images and their corresponding 14-label targets.



The `model.py` script defines a DenseNet121 model initialized with pretrained ImageNet weights and modified for 14-label chest X-ray classification.



The model uses a multilabel classification setup with binary cross-entropy loss.



\## 3. Single-GPU Benchmark



The benchmark was executed on NERSC Perlmutter using one NVIDIA A100-SXM4-40GB GPU.



\*\*Benchmark configuration:\*\*



| Parameter | Value |

|---|---|

| Model | DenseNet121 |

| GPU used | 1 NVIDIA A100 40GB |

| Training images | 4,096 |

| Validation images | 128 |

| Epochs | 1 |

| Batch size | 16 |

| DataLoader workers | 4 |

| Learning rate | 0.0001 |

| Optimizer | AdamW |

| Precision | Mixed precision |



\*\*Benchmark results:\*\*



| Metric | Result |

|---|---|

| Training loss | 0.2022 |

| Training time | 17.99 seconds |

| Training throughput | \*\*227.63 images/second\*\* |

| Validation mean AUROC | 0.6579 |

| Slurm job ID | 61900136 |

| Slurm job status | COMPLETED |



The throughput measures the training loop, excluding validation, model initialization, and other job overhead.



The validation AUROC is preliminary because only 128 validation images were evaluated. This benchmark is intended primarily to establish training throughput rather than final classification performance.



\## 4. Running the Benchmark



The benchmark is defined in `benchmark\_1gpu.py` and submitted through `benchmark\_1gpu.slurm`.



On Perlmutter, from the project root:



```bash

sbatch preprocessing/benchmark\_1gpu.slurm

```



The script expects the dataset and pretrained weights to be available under:



```text

/pscratch/sd/p/pravch1/chestxray\_project/

```



The preprocessed images are stored in `processed\_images/`, and the dataset split manifests are stored in `manifests/`.



\## 5. Results and Reproducibility



The following benchmark outputs are included in this folder:



\- `bench\_61900136.out` — benchmark execution log and training metrics.

\- `validation\_auc.csv` — validation AUROC values for individual disease labels.



The trained model checkpoint is stored on Perlmutter and is not included in the GitHub repository.



The full image dataset is also stored on Perlmutter rather than GitHub because of its size.



\## 6. Next Steps



The preprocessed dataset and single-GPU benchmark provide the foundation for subsequent experiments comparing distributed training configurations and rarity-aware sampling strategies.



Future experiments will evaluate multi-GPU training throughput, scaling efficiency, and classification performance for rare disease labels.

