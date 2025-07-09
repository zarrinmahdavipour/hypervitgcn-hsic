# HyperViTGCN: Hyperspectral Image Classification

This repository contains the implementation of **HyperViTGCN**, a hybrid model for hyperspectral image classification combining 3D-CNNs, lightweight Vision Transformers, and attention-guided GCNs. The model is evaluated on Indian Pines, PaviaU, Houston2013, and Botswana datasets, achieving state-of-the-art performance with reduced computational complexity.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hypervitgcn/hsic.git
   cd HyperViTGCN
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets (Indian Pines, PaviaU, Houston2013, Botswana) and place them in the `datasets/` folder.

## Usage

1. Configure hyperparameters in `config.yaml`.
2. Run the main script:
   ```bash
   python main.py --config config.yaml
   ```

## Datasets

- **Indian Pines**: 145x145 pixels, 200 bands, 16 classes.
- **PaviaU**: 610x340 pixels, 103 bands, 9 classes.
- **Houston2013**: 349x1905 pixels, 144 bands, 15 classes.
- **Botswana**: 1476x256 pixels, 145 bands, 14 classes.

## Results

- **Indian Pines**: 94.3% OA
- **PaviaU**: 96.0% OA
- **Houston2013**: 93.7% OA
- **Botswana**: 96.4% OA

## Citation

Please cite our paper if you use this code:
```bibtex
@article{mahdavipour2025hypervitgcn,
  title={Hyperspectral Image Classification Using a Hybrid Vision Transformer and Graph Convolutional Network},
  author మీరు ఈ కోడ్‌ను ఉపయోగిస్తే, దయచేసి మా పేపర్‌ను ఉదహరించండి:
```bibtex
@article{mahdavipour2025hypervitgcn,
  title={Hyperspectral Image Classification Using a Hybrid Vision Transformer and Graph Convolutional Network},
  author={Mahdavipour, Zarrin and Xiao, Liang and Yang, Jingxiang and Farooque, Ghulam},
  journal={Journal of LaTeX Class Files},
  volume={14},
  number={8},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.