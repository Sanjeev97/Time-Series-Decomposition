# TSDecomposition: Revisiting the Decomposition for Time Series Forecasting

A PyTorch implementation of decomposition-enhanced time series forecasting models that improve state-of-the-art architectures through strategic time series decomposition techniques.

## 📋 Overview

This repository contains the implementation of our paper "Revisiting the Decomposition for Time Series Forecasting", which introduces a novel approach to enhance existing time series forecasting models through decomposition techniques. Our method achieves ~10% average MSE reduction across multiple state-of-the-art baselines while maintaining computational efficiency.

### Key Features

- **Decomposition-Enhanced Models**: DR variants of popular models (DRDLinear, DRPatchTST, DRTimesNet, DRiTransformer)
- **Novel Dual-MLP Models**: RMSM and RMM models with linear time complexity O(L)
- **Multiple Decomposition Methods**: Moving Average, Mixture of Experts, and Frequency-based decomposition
- **Comprehensive Evaluation**: Tested on benchmark datasets plus real-world hydrological data

## 🏗️ Architecture

```
Input Time Series X
       ↓
[Moving Average Decomposition]
       ↓
   Trend XT ←→ Seasonal XS
       ↓              ↓
[RevIN + MLP]    [Backbone Model]
       ↓              ↓
   Prediction YT + Prediction YS
       ↓
   Final Output Y
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/username/TSDecomposition.git
cd TSDecomposition
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train DRDLinear on ETTh2 dataset
python run.py \
  --is_training 1 \
  --model DRDLinear \
  --data ETTh2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 32

# Train RMSM model
python run.py \
  --is_training 1 \
  --model RMSM \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --seq_len 96 \
  --pred_len 192
```

### Using Shell Scripts

```bash
# Run comprehensive experiments
chmod +x scripts/DRDLinear/ETTh2_96_input.sh
./scripts/DRDLinear/ETTh2_96_input.sh
```

## 📊 Datasets

### Benchmark Datasets
- **ETT (Electricity Transformer Temperature)**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: 21 meteorological indicators from Max Planck Institute
- **Electricity**: Electricity consumption from 321 clients (2012-2014)

### Hydrological Dataset
- **Source**: USGS river stream gauges from Comite River, Louisiana
- **Stations**: 4 monitoring stations (07377600, 07377760, 07378050, 07380120)
- **Period**: 2015-2023, 15-minute intervals
- **Variables**: Stream gauge heights in inches

## 🎯 Models

### Decomposition-Enhanced Models (DR Series)
- **DRDLinear**: Decomposition + Reversible Instance Normalization + Linear layers
- **DRPatchTST**: Decomposition + PatchTST backbone
- **DRTimesNet**: Decomposition + TimesNet backbone  
- **DRiTransformer**: Decomposition + iTransformer backbone

### Novel Dual-MLP Models
- **RMSM (RevIN MLP Shift-MLP)**: Uses shift forecasting for seasonal components
- **RMM (RevIN MLP MLP)**: Standard 3-layer MLP for both components

### Supported Backbone Models
- iTransformer
- PatchTST
- TimesNet
- DLinear

## 🔧 Key Parameters

```bash
# Core parameters
--model            # Model name: DRDLinear, DRPatchTST, DRTimesNet, DRiT, RMSM, RMM
--seq_len          # Input sequence length (default: 96)
--pred_len         # Prediction horizon: 96, 192, 336, 720
--moving_avg       # Moving average window size (default: 25)
--decomp_method    # Decomposition method: moving_avg, dft_decomp
--d_model_trend    # Dimension for trend MLP (default: 512)

# Dataset parameters
--data             # Dataset: ETTh1, ETTh2, ETTm1, ETTm2, custom
--root_path        # Dataset root directory
--data_path        # Dataset file name
--features         # Forecasting task: M, S, MS
```

## 📈 Results

### Performance Improvements
| Model | MSE Reduction | MAE Reduction |
|-------|---------------|---------------|
| iTransformer | 6.81% | 9.10% |
| PatchTST | 13.62% | 12.39% |
| TimesNet | 9.95% | 9.35% |
| DLinear | 15.82% | 11.94% |

### Computational Efficiency
| Model | Time Complexity | Training Time | Inference Time |
|-------|----------------|---------------|----------------|
| RMSM/RMM | O(L) | 10ms | 5ms |
| DLinear | O(L) | 9ms | 4ms |
| PatchTST | O(L²/P²) | 20ms | 11ms |
| TimesNet | O(L²) | 136ms | 75ms |

## 🗂️ Project Structure

```
TSDecomposition/
├── data/                    # Data loading and preprocessing
│   ├── data_factory.py     # Data factory for different datasets
│   └── data_loader.py      # Dataset classes
├── models/                  # Model implementations
│   ├── DRDLinear.py        # Decomposition-enhanced DLinear
│   ├── DRPatchTST.py       # Decomposition-enhanced PatchTST
│   ├── DRTimesNet.py       # Decomposition-enhanced TimesNet
│   ├── DRiTransformer.py   # Decomposition-enhanced iTransformer
│   ├── RMSM.py             # RevIN MLP Shift-MLP model
│   └── RMM.py              # RevIN MLP MLP model
├── layers/                  # Neural network components
│   ├── Decomposition.py    # Decomposition methods
│   ├── MLPs.py             # MLP layers and RevIN
│   ├── SelfAttention_Family.py # Attention mechanisms
│   └── embedding.py        # Embedding layers
├── exp/                     # Experiment management
├── utils/                   # Utility functions
├── scripts/                 # Training scripts
└── run.py                   # Main training script
```

## 🧪 Experiments

### Run Single Experiment
```bash
python run.py \
  --is_training 1 \
  --model_id test_run \
  --model DRDLinear \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --train_epochs 100 \
  --patience 20
```

### Ablation Studies
```bash
# Test different decomposition methods
python run.py --decomp_method moving_avg --model DRiT --data Weather
python run.py --decomp_method dft_decomp --model DRiT --data Weather
```

### Batch Experiments
```bash
# Run all prediction horizons for Weather dataset
for pred_len in 96 192 336 720; do
  python run.py --model RMSM --data custom --pred_len $pred_len
done
```

## 📋 Requirements

```
torch==1.11.0
numpy==1.21.6
pandas==1.5.3
matplotlib==3.7.0
scikit-learn==1.5.1
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- Thanks to the authors of iTransformer, PatchTST, TimesNet, and DLinear for their foundational work
- USGS for providing the hydrological dataset
- The time series forecasting community for benchmark datasets


**Note**: This implementation is part of our research on decomposition-enhanced time series forecasting. The code is provided for reproducibility and further research.
