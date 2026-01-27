# Stage 3 local run (patched v3)

This folder contains:

- `CSE400_Final_Draft_V5_local_patched_clean_v3.ipynb` – CLEAN notebook (outputs cleared) + robust loss compile + auto-detects `csv/` subfolder
- `train_stage3_local_patched_v3.py` – runnable script version (CLI) + same robustness + auto-detects `csv/` subfolder
- `README_local_run_v3.md` – this file

## Notebook quick start (Windows)

1. Activate your venv:
   - PowerShell: `\.venv\Scripts\Activate.ps1`

2. Set your dataset path.

   You can point this to either:
   - the folder that *directly contains* `conditions.csv`, `encounters.csv`, etc, **or**
   - a parent folder that contains a `csv\` subfolder (v3 will auto-detect).

   PowerShell example:
   ```powershell
   $env:SYNTHEA_CSV_DIR="D:\CSE400 Final Defence Draft\SyntheaMass Data"
   ```

3. Launch Jupyter:
   - `jupyter lab`

4. Open `CSE400_Final_Draft_V5_local_patched_clean_v3.ipynb`

5. In the notebook menu: **Kernel → Restart & Run All**

## Script quick start

```powershell
\.venv\Scripts\Activate.ps1
python train_stage3_local_patched_v3.py --data_dir "D:\CSE400 Final Defence Draft\SyntheaMass Data" --output_dir "output_stage3" --max_patients 50000
```

Quick smoke test:

```powershell
python train_stage3_local_patched_v3.py --data_dir "D:\CSE400 Final Defence Draft\SyntheaMass Data" --output_dir "output_stage3" --max_patients 5000 --epochs 3 --batch_size 16 --target_top_k 15 --min_label_count 50
```

## GPU note (Windows)

On native Windows, recent TensorFlow versions are usually CPU-only (you may see `TF sees GPUs: []`).

If you need NVIDIA CUDA GPU training, prefer running TensorFlow in **WSL2 (Ubuntu)**, or consider the DirectML plugin (with its own version constraints).
