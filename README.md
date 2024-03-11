# Sampling-based Approximation of the Generalized Gauss-Newton Matrix

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/github/license/f-lair/sampling-based-ggn-approximation)

This work was done as a research project in the Methods of Machine Learning group at the University of Tübingen.

The goal was to investigate methods to approximate the [Generalized Gauss-Newton (GGN) matrix](https://aleximmer.github.io/assets/immer_msc_thesis.pdf), a positive semi-definite approximation to the often indefinite Hessian of deep neural networks.
While second-order information can be helpful to improve training or do uncertainty quantification, the computation of the GGN is still a hurdle due to the summation over the entire dataset.
Here, we try to approximate the GGN in the data dimension.


## Installation

To run the scripts in this repository, **Python 3.10** is needed.
Then, simply create a virtual environment and install the required packages via

```bash
pip install -r requirements.txt
```

## Usage

The python script `src/train.py` is used to train a deep neural network and store trained parameters on disk.
Then, the script `src/run_experiment.py` can be used to run one of three experiments (frobenius, eigen, laplace).
Both implement a command line interface that allows specifying the model architecture, sampling strategy and several other hyperparameters.
**Important:** Scripts must be run inside the directory `src/`.
For the details, invoke the scripts with the flag `-h`/`--help`.