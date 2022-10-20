# p3VAE

This repository contains code for the paper:

R. Thoreau, L. Risser, V. Achard, B. Berthelot and X. Briottet, "p3VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images", 2022.

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

## Feedback

Please send any feedback to romain.thoreau@onera.fr

## Bibtex citation

```
@misc{https://doi.org/10.48550/arxiv.2210.10418,
  doi = {10.48550/ARXIV.2210.10418},
  url = {https://arxiv.org/abs/2210.10418},
  author = {Thoreau, Romain and Risser, Laurent and Achard, Véronique and Berthelot, Béatrice and Briottet, Xavier},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences, I.2.6; I.2.10, 68T45},
  title = {p$^3$VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
