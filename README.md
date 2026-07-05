# MPNet
We provide our PyTorch implementation of reference-guided image dehazing to explore the impact of reference content relevance on image dehazing.

## 1. Environment & Installation
### 1.1 Environment Versions
All experiments are conducted with the following software versions:
- Python: `3.8`
- CUDA: `11.3`
- PyTorch: `1.12.0`
- cuDNN: `8.3.2`

### 1.2 Installation Steps
1. Clone this repository:
```bash
git clone https://github.com/liu-xiaoliu/MPNet.git
cd MPNet
```

2. Create and activate a conda environment:
```bash
conda create -n mpnet python=3.8
conda activate mpnet
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 2. Datasets
Our experiments are conducted on the following publicly available datasets:
- **[O-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/)**: Real-world outdoor dehazing benchmark from NTIRE 2018
- **[NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/)**: Non-homogeneous dehazing benchmark from NTIRE 2020
- **[BeDDE](https://github.com/xiaofeng94/BeDDE-for-defogging)**: Real-world dehazing evaluation benchmark dataset
- **[RESIDE](https://sites.google.com/view/reside-dehaze-datasets)**: Comprehensive synthetic dehazing benchmark dataset

---

## 3. Quick Start (Command Examples)
### 3.1 Evaluation / Test
The pretrained checkpoint trained on the BeDDE dataset is included in this repository under the `checkpoints/` directory, which can reproduce the main experimental results reported in the paper.

Run the official test script to reproduce the reported metrics:
```bash
python test.py --dataroot [path to your dataset]
```

### 3.2 Training
To train the model from scratch:
```bash
python train.py --dataroot [path to your dataset]
```
All hyperparameters and configuration options can be modified in the `options/` directory.

---

## 4. Repository Structure
```
MPNet/
├── LICENSE             # Open source license
├── README.md           # This file
├── requirements.txt    # Dependencies with version numbers
├── data/               # Dataset loading logic
├── models/             # Core model architecture files
├── options/            # Training & test configuration options
├── util/               # Utility functions
├── checkpoints/        # Pretrained model weights
├── test.py             # Test script
└── train.py            # Training script
```

---

## 5. License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{mpnet2026,
  title   = {Mutual-Prompting for Reference-Guided Image Dehazing: Exploring Reference Content Relevance},
  author  = {Liu, Yanting and Yin, Hui and Yang, Ying and Chong, Aixin},
  year    = {2026}
}
```

---

## Acknowledgments
Our code is developed based on the [contrastive-unpaired-translation (CUT)](https://github.com/taesungp/contrastive-unpaired-translation) repository. We thank the original authors for their excellent open-source work.

---

## Contact
For any questions, please contact: 19112024@bjtu.edu.cn
