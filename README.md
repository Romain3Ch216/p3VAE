# Physics-informed Variational Autoencoders for Improved Robustness to Environmental Factors of Variation

This repository contains code for the following paper, under review in Springer Machine Learning: <br>
[Physics-informed Variational Autoencoders for Improved Robustness to Environmental Factors of Variation](https://arxiv.org/abs/2210.10418)

A short version of this work has been accepted as a workshop paper at [Machine Learning for Remote Sensing, ICLR 2024](https://ml-for-rs.github.io/iclr2024/camera_ready/papers/17.pdf).

Please cite this paper if you use the code in this repository as part of a published research project (see bibtex citation below).

## Extrapolation of damped pendulum time series

<p align="center">
  <img src="https://github.com/Romain3Ch216/p3VAE/blob/main/pendulum_extrapolation.png" alt="Pendulum extrapolation">
</p>

In order to reproduce the results, you can train the networks with the scripts `exps/extrapolation/train.sh` (baselines) and `exps/extrapolation/p3vae_train.sh` (our method).

## Hyperspectral image classification

<p align="center">
  <img src="https://github.com/Romain3Ch216/p3VAE/blob/main/land_cover_classification.png" alt="Hyperspectral image classification">
</p>

The airborne hyperspectral images acquired during the CAMCATT-AI4GEO experiment in Toulouse, France are publicly available here: https://camcatt.sedoo.fr/

To load and save image patches, use an instance of the `GeoDataset` class in the `exps/classification/data.py` file.

In order to reproduce the results, you can train the networks by running `exps/classification/train.py` with default arguments.

## Methane plume inversion

<p align="center">
  <img src="https://github.com/Romain3Ch216/p3VAE/blob/main/methane_inversion.png" alt="Methane inversion">
</p>

Scripts to run the optimal estimation algorithm and apply p$^3$VAE to the inversion of methane plume from hyperspectral satellite data are in the `exps/inversion` folder.

## Setup

The code was run using python 3.8:

1. create a python [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
2. clone this repo: ```git clone https://github.com/Romain3Ch216/p3VAE.git```
3. navigate to the repository: ```cd p3VAE```
4. install python requirements: ```pip install -r requirements.txt```

## Feedback

Please send any feedback to romain.thoreau@cnes.fr, or open an issue.

## Bibtex citation

```
@article{thoreau2022p,
  title={p $\^{} 3$ VAE: a physics-integrated generative model. Application to the pixel-wise classification of airborne hyperspectral images},
  author={Thoreau, Romain and Risser, Laurent and Achard, V{\'e}ronique and Berthelot, B{\'e}atrice and Briottet, Xavier},
  journal={arXiv preprint arXiv:2210.10418},
  year={2022}
}

```
