from setuptools import setup, find_packages

setup(
    name="anomaly_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
        'PyYAM',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'pandas',
        'seaborn',
        'tqdm',
        'opencv-python',
        'Pillow',
        'scikit-image',
        'albumentations',
        'tensorboard',
        'tensorboardX'
    ],
)