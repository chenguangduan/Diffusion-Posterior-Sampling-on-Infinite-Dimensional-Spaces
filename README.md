# Diffusion Posterior Sampling on Infinite-Dimensional Spaces

PyTorch implementation of diffusion-based posterior sampling for Bayesian inverse problems with a prior learned from samples. The example recovers a spatially varying coefficient in an elliptic equation from noisy observations and compares diffusion posterior sampling with importance sampling.

## Installation

```bash
python -m pip install torch numpy matplotlib
```

## Run the example

Run the following commands from the repository root. Pretrained checkpoints for both drift operators are included in `checkpoints/`.

```bash
# Sample with A = I
python main.py --mode sample

# Sample with A = \gamma I - \Delta
python main.py --mode sample --gamma 1.0

# Compare both diffusion estimates with importance sampling
python main.py --mode plot --gamma 1.0
```

The sampling commands save `diffusion_sample.txt` and `diffusion_sample_1.txt`. The plotting command requires both files, computes the importance-sampling baseline, and saves the comparison to `result_1.pdf`.

## Train the prior models (optional)

```bash
python main.py --mode train
python main.py --mode train --gamma 1.0
```

Training saves the models in `checkpoints/`. Hyperparameters are defined in `main.py`. CUDA is used automatically when available; otherwise, the code runs on CPU.
