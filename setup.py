from setuptools import setup, find_packages
from pathlib import Path


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='mini-esm',
    version='1.0',
    author='Kahiri Abidi',
    author_email='khairi.abidi@majesteye.com',
    description='A short description of your project',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abidikhairi/evolutionary-scale-modeling',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pytorch-lightning>=2.3.3',
        'torch>=2.3.1',
        'torchmetrics>=1.4.0.post0',
        'torchtext>=0.18.0',
        'torchvision>=0.18.1',
        'datasets>=2.20.0',
        'tokenizers>=0.19.1',
        'pandas>=1.5.3',
        'wandb>=0.17.5',
        'transformers>=4.42.4'
    ]
)
