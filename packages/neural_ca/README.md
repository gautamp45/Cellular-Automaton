# neural_ca

PyTorch implementation of Growing Neural Cellular Automata — a self-organising system
where cells learn to grow into a target image and regenerate after damage.

## Structure

```
neural_ca/
├── src/
│   ├── model.py          # NCA model (perception + stochastic update rule)
│   ├── helper.py         # Loss functions, image utilities, seed/mask generators
│   ├── train.py          # Training loop
│   ├── visualize.py      # Animation renderer
│   └── medmnist_loader.py
├── configs/
│   ├── train_config.yaml
│   └── visualize_config.yaml
├── data/                 # Target images (gitignored)
├── models/               # Saved .pt checkpoints (gitignored)
├── animations/           # Output GIFs (gitignored)
└── notebooks/
    └── Growing_Neural_Cellular_Automata.ipynb
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Edit `configs/train_config.yaml` to set the target image and hyperparameters, then:

```bash
cd packages/neural_ca
python src/train.py
```

All paths in the config are resolved relative to the `neural_ca/` package directory.

## Visualisation

After training, render a growth animation:

```bash
cd packages/neural_ca
python src/visualize.py
```

Requires ImageMagick for GIF export (`brew install imagemagick`).

## Supported filters

`sobel` · `scharr` · `gaussian` · `laplacian` · `mean`

## Supported loss functions

`L1` · `L2` · `Manhattan` · `Hinge`
