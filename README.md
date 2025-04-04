# Autoencoder-Based Anomaly Detection System

This project implements a comprehensive anomaly detection system using multiple autoencoder architectures in PyTorch. The system is designed to detect anomalies in images through reconstruction error analysis.

## Features

- Multiple autoencoder architectures:
  - CNN-based autoencoder
  - ResNet-based autoencoder
  - Vision Transformer (ViT) autoencoder
  - Diffusion model-based autoencoder
- Modular codebase with separate training and inference pipelines
- Comprehensive training monitoring with TensorBoard
- Detailed anomaly analytics and visualization
- Performance comparison between different architectures

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gustavorubim/anomaly-detection.git
   cd anomaly-detection
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate anomaly_detection
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Training

To train a model:
```bash
python anomaly_detection/scripts/train.py --config config/default.yaml
```

Training options can be modified in the config file or by passing command line arguments.

### Inference

To run inference with a trained model:
```bash
python anomaly_detection/scripts/inference.py --model path/to/model --output results
```

This will:
1. Process the test set
2. Generate visualizations of anomalies
3. Create a report with evaluation metrics

### TensorBoard

To monitor training:
```bash
tensorboard --logdir logs
```

## Project Structure

```
anomaly_detection/
├── config/             # Configuration files
├── data/               # Data loading and preprocessing
│   ├── preprocessing.py  # Image normalization, resizing, grayscale
│   ├── augmentation.py   # Data augmentation transforms
│   └── dataset.py       # PreprocessingPipeline implementation
├── models/             # Autoencoder implementations
├── training/           # Training pipeline
├── inference/          # Inference and evaluation
├── visualization/      # Visualization tools
├── utils/              # Utility functions
├── scripts/            # Executable scripts
└── notebooks/          # Jupyter notebooks for analysis
```

## Data Preprocessing

The system includes a comprehensive `PreprocessingPipeline` that handles:

- Image normalization (to [0,1] or custom mean/std)
- Resizing to target dimensions
- Grayscale conversion (optional)
- Data augmentation (training only)
- Configurable through YAML files

Example configuration:
```yaml
preprocessing:
  target_size: [256, 256]
  grayscale: false
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  augmentations:
    flip: true
    rotation: 15
    brightness: 0.2
    contrast: 0.2
```

## Dataset

This project uses the MVTec Anomaly Detection dataset. Place the dataset in the `data` directory with the following structure:

```
data/
└── zipper/
    ├── train/
    │   └── good/       # Normal training images
    ├── test/
    │   ├── good/       # Normal test images
    │   └── defect/     # Anomalous test images
    └── ground_truth/   # Ground truth masks for anomalies
```

## Results

The system generates several outputs:
- Model checkpoints in `checkpoints/`
- Training logs in `logs/`
- Inference results in `results/`
  - Visualizations of anomalies
  - Evaluation reports
  - Statistical analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.