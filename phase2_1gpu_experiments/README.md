Phase 2: Single-GPU Rarity-Aware Sampling Experiments

Objective

Evaluate whether rarity-aware sampling improves validation AUROC for rare findings in multi-label chest X-ray classification, compared with standard shuffled sampling, before scaling to multi-GPU training.

Dataset and model

\- Dataset: NIH ChestX-ray14, 14 multi-label findings. The project uses predefined training and validation manifests under manifests/ and preprocessed images under processed\_images/.

\- Model: DenseNet-121 initialized from local ImageNet weights (pretrained/densenet121-a639ec97.pth), with its classifier replaced by a 14-output linear layer.

\- Inputs: Images are converted to three-channel grayscale tensors and normalized with ImageNet mean and standard deviation (see ../preprocessing/dataset.py).

\- Loss and optimization: BCEWithLogitsLoss, AdamW, base learning rate 1e-4, linear learning-rate scaling (effective LR 1e-4 for one GPU), one warmup epoch, automatic mixed precision.

\- Hardware: One NVIDIA A100 80 GB GPU on NERSC Perlmutter, using torchrun with one process; batch size 64, 8 data-loader workers.

Experimental design

Twelve full-data runs: four samplers × three random seeds (0, 1, 2). Each run trained for 8 epochs, with MAXTRAIN=0 and MAXVAL=0 (no subset limits).

Sampler	Configuration	Rare-group images targeted per batch

Default	SAMPLER=default	No enforced quota

Rarity-aware RPB4	SAMPLER=rarity,RPB=4	4 of 64

Rarity-aware RPB8	SAMPLER=rarity,RPB=8	8 of 64

Rarity-aware RPB16	SAMPLER=rarity,RPB=16	16 of 64





The rarity-aware sampler defines Edema, Fibrosis, Pneumonia, and Hernia as its rare group. An image is eligible for the rare pool if it has at least one of these labels; other images form the common pool. It shuffles the pools each epoch and resizes/repeats sampled indices as needed. Thus, oversampling can repeat examples; the last partial batch may not meet the exact quota. Emphysema is not in the sampler's designated rare group, even though it is among the less frequent labels in the dataset.

Results

Results are computed across seeds using the best validation macro-AUROC in each run. Macro-AUROC is the mean of available per-disease AUROCs; the best checkpoint and per-disease CSV correspond to the epoch with the highest validation macro-AUROC, not necessarily epoch 8. Standard deviations below are sample standard deviations across three seeds.

Sampling method	Seed 0	Seed 1	Seed 2	Mean best AUROC	Std. dev.	Training GPU-hours (3 runs)

Default	0.8266	0.8328	0.8323	0.8305	0.0034	0.9805

RPB4	0.8254	0.8262	0.8264	0.8260	0.0005	0.8929

RPB8	0.8198	0.8164	0.8216	0.8193	0.0026	0.9828

RPB16	0.7909	0.7886	0.7925	0.7907	0.0019	0.9137





Main finding: Default sampling had the highest overall validation AUROC. RPB4 was close (0.0045 lower), while more aggressive rare-image quotas reduced macro-AUROC, especially RPB16. These results do not demonstrate a consistent overall benefit from rarity-aware sampling.

Rare-disease validation AUROC

Mean per-disease AUROC across three seeds, evaluated at each run's best macro-AUROC checkpoint:

Disease	Default	RPB4	RPB8	RPB16

Hernia	0.8768	0.8658	0.8865	0.8387

Pneumonia	0.7287	0.7299	0.7218	0.7110

Fibrosis	0.8101	0.8116	0.7914	0.7658

Edema	0.9176	0.9209	0.9097	0.9132

Emphysema (not sampler-targeted)	0.8828	0.8762	0.8708	0.8106





RPB4 gave small gains for Pneumonia, Fibrosis, and Edema; RPB8 improved Hernia. These improvements were not uniform across diseases, and the validation set has relatively few positives for some findings (particularly Hernia). Do not interpret small differences as statistically established improvements.

GPU accounting

\- 3.7699 GPU-hours is the sum of gpu\_hours in the 12 history CSVs. The training code measures synchronized training-loop time only, multiplied by world size (one GPU). This excludes validation, initialization, checkpoint writing, and other overhead.

\- The 12 Slurm jobs together elapsed 4 h 17 min 33 sec. Their allocations reported four A100 GPUs per job even though torchrun --nproc\_per\_node=1 trained on one GPU; consequently the corresponding reserved capacity is approximately 17.17 allocated GPU-hours (4 × elapsed time). This is not the same as training GPU-hours or NERSC's account-specific charging units.

\- Per-epoch time and throughput are in results/history\_\*.csv. gpu\_utilization/, gpu\_accounting/, and logs/ are excluded from this repository folder by .gitignore.

Reproducing the runs on Perlmutter

The repository's ../preprocessing/dataset.py and ../preprocessing/model.py provide the data loader and DenseNet-121 model. The runtime project root must contain:

chestxray\_project/

├── manifests/train.csv, val.csv, test.csv

├── processed\_images/

├── pretrained/densenet121-a639ec97.pth

├── preprocessing/dataset.py, model.py

└── phase2\_1gpu\_experiments/

&#x20;   ├── run\_1gpu.sh

&#x20;   ├── scripts/train\_ddp.py, rarity\_sampler.py, check\_sampler.py, eval\_metrics.py

&#x20;   └── results/

The code currently uses an absolute Perlmutter project path in ../preprocessing/dataset.py, and run\_1gpu.sh also hardcodes the project directory and Slurm account (m4431\_g). Adjust these for another environment. train\_ddp.py imports dataset and model directly; ensure the preprocessing directory is available on PYTHONPATH (or otherwise importable) in the launch environment. Install/use the required PyTorch, torchvision, NumPy, pandas, Pillow, and scikit-learn packages; on Perlmutter the job script loads the pytorch module. Ensure the Slurm logs/ and gpu\_utilization/ directories exist before submission.



From phase2\_1gpu\_experiments/, a full default run is submitted with:

sbatch --export=ALL,SAMPLER=default,EPOCHS=8,SEED=0,MAXTRAIN=0,MAXVAL=0,BATCH=64,WORKERS=8,WARMUP=1 run\_1gpu.sh

For rarity-aware runs, set SAMPLER=rarity and RPB=4, 8, or 16. Repeat each configuration with SEED=0, 1, and 2 (12 jobs total). For example:

sbatch --export=ALL,SAMPLER=rarity,RPB=4,EPOCHS=8,SEED=0,MAXTRAIN=0,MAXVAL=0,BATCH=64,WORKERS=8,WARMUP=1 run\_1gpu.sh

The launch script uses torchrun --standalone --nproc\_per\_node=1 and logs GPU utilization with nvidia-smi. Run jobs sequentially or manage dependencies if needed. Note that the original Slurm resource request reserves more GPUs than the one used for training; adjust the allocation to your site's policy when reproducing.

Files and interpretation notes

\- results/history\_1gpu\_\*.csv: epoch-level loss, training seconds, images/sec, training GPU-hours, and validation macro-AUROC.

\- results/validation\_auc\_1gpu\_\*.csv: per-disease validation AUROC from the run's best macro-AUROC checkpoint.

\- scripts/: exact verified copies of the scripts used on Perlmutter.

\- summarize\_results.py: aggregates best AUROC and training GPU-hours from history CSVs (python phase2\_1gpu\_experiments/summarize\_results.py from repository root).

\- Model checkpoints, processed images, raw logs, and the transfer archive are not included.

Limitations: These are validation-set comparisons, not held-out test results. Only three seeds were used, with one training duration (8 epochs), one model, and one dataset split. Rare-label validation counts are small; differences may reflect sampling variance. Repeating rare examples changes the effective training distribution and can degrade other labels. No claim of improved clinical performance or multi-GPU scaling is made from these single-GPU results.

