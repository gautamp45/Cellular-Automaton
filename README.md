# Cellular Automaton Monorepo

Three independent Python packages exploring different facets of Cellular Automata —
from PyTorch self-organisation to stream ciphers to interactive Life-like simulations.

## Packages

### [`packages/neural_ca`](packages/neural_ca/)

**Growing Neural Cellular Automata** — PyTorch implementation where a small neural
network learns the local update rule that causes a grid of cells to grow into a target
image and regenerate after damage.

Key files: `model.py` · `train.py` · `visualize.py` · `helper.py`  
Run: `cd packages/neural_ca && python src/train.py`

---

### [`packages/ca_cryptography`](packages/ca_cryptography/)

**1D CA Stream Cipher** — A 1D elementary CA evolves a random initial state; the final
state is XOR-ed with a binary-encoded plaintext to produce ciphertext. Includes a
brute-force reverse-evolution attack demo.

Key files: `ca.py` · `rsrg.py` · `enc_dec.py` · `main.py`  
Run: `cd packages/ca_cryptography/src && python main.py`

---

### [`packages/classical_ca`](packages/classical_ca/)

**Classic CA simulations** — Conway's Game of Life, general Life-like 2D CA with a
Pygame interactive UI (draw cells, toggle rules), 1D elementary CA, population
graphing, and a dataset builder for rule-prediction experiments.

Key files: `ca_2d.py` · `ca_1d.py` · `game_of_life.py` · `main_run.py`  
Run: `cd packages/classical_ca/src && python main_run.py`

---

## Repository layout

```
packages/
├── neural_ca/
│   ├── src/              # model, helper, train, visualize
│   ├── configs/          # train_config.yaml, visualize_config.yaml
│   ├── notebooks/
│   └── data/             # target images (gitignored)
├── ca_cryptography/
│   └── src/              # ca, rsrg, enc_dec, converter, visualiser, main, attacks
└── classical_ca/
    ├── src/              # ca_1d, ca_2d, game_of_life, game_board, grapher, ...
    └── notebooks/        # Random Forest rule predictor (WIP)
```

## Quick start

Each package has its own `requirements.txt`:

```bash
pip install -r packages/neural_ca/requirements.txt
pip install -r packages/ca_cryptography/requirements.txt
pip install -r packages/classical_ca/requirements.txt
```
