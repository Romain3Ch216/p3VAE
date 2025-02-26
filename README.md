# p3VAE

This repository contains code for the following paper, under review in Springer Machine Learning: <br>
[Physics-informed Variational Autoencoders for Improved Robustness to Environmental Factors of Variation](https://arxiv.org/abs/2210.10418)

A short version of this work has been accepted as a workshop paper at [Machine Learning for Remote Sensing, ICLR 2024](https://ml-for-rs.github.io/iclr2024/camera_ready/papers/17.pdf).

Please cite this paper if you use the code in this repository as part of a published research project (see bibtex citation below).

## Setup

The code was run using python 3.8:

1. create a python [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
2. clone this repo: ```git clone https://github.com/Romain3Ch216/p3VAE.git```
3. navigate to the repository: ```cd p3VAE```
4. install python requirements: ```pip install -r requirements.txt```

## Reproducing The Results

We provide the data and code that were used to compute results from experiments of section 5.
The `train.py` script was used to train the models which weights are in the `results` folder. 
Other files were used to plot the figures of section 5.

For instance, to reproduce the figure 7 of section 5 for the p3VAE with seed 103, run the following script:

```python max_likelihood_estimate.py './results/p3VAE/103'```

The figure will be saved in the './results/p3VAE/Figures` folder.

## Loading real data

The airborne hyperspectral images acquired during the CAMCATT-AI4GEO experiment in Toulouse, France are publicly available here: https://camcatt.sedoo.fr/

To load and save image patches, use an instance of the `GeoDataset` class in the `data.py` file.

## Feedback

Please send any feedback to romain.thoreau@cnes.fr

## Bibtex citation

```
@article{thoreau2022p,
  title={p $\^{} 3$ VAE: a physics-integrated generative model. Application to the pixel-wise classification of airborne hyperspectral images},
  author={Thoreau, Romain and Risser, Laurent and Achard, V{\'e}ronique and Berthelot, B{\'e}atrice and Briottet, Xavier},
  journal={arXiv preprint arXiv:2210.10418},
  year={2022}
}

```
