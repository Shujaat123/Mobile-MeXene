# Mobile-MeXene

This repository contains the implementation of the manuscript:

> **Optimizing Spectral Prediction in MXene-Based Metasurfaces Through Multi-Channel Spectral Refinement and Savitzky-Golay Smoothing**

---

## 📌 Overview

This work focuses on making spectral prediction from MXene-based metasurface images using:

- Deep neural network with MobileNetV2 backbone  
- **Multi-Channel Spectral Refinement (MCSR)** module  
- **Savitzky–Golay smoothing** applied as post-processing at inference 

## 🏗️ Model Architecture

Image → Backbone → FC Projection → Spectrum
↓
Multi-Channel Spectral Refinement
↓
Savitzky–Golay smoothing
↓
Output Spectrum

---

## 🚀 Training

Please refer to the argparser in train.py file for arguments

## 📊 Outputs

Per Run

{
run_xx/
├── base_split/
├── trainpct_xxx/
│   ├── epoch_metrics.csv
│   ├── history.json
│   ├── test_metrics_raw.json
│   ├── test_metrics_savgol_grid.csv
│   └── checkpoints/
}

Global

{
all_runs_summary.csv
all_runs_smoothing_summary.csv
aggregate_metrics.csv
aggregate_savgol_metrics.csv
}

---

## 📜 Citation

If you use this code, please cite:

Khan, S., & Waseer, W. I. (2026). Optimizing Spectral Prediction in MXene-Based Metasurfaces Through Multi-Channel Spectral Refinement and Savitzky-Golay Smoothing. arXiv preprint arXiv:2602.08406.



