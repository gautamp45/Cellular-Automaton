# classical_ca

Classic Cellular Automata implementations: Conway's Game of Life, Life-like 2D CA with
an interactive Pygame UI, 1D elementary CA, and dataset tooling for rule prediction.

## Structure

```
classical_ca/
├── src/
│   ├── ca_1d.py           # 1D elementary CA — single-cell seed, standard visualisation
│   ├── ca_2d.py           # 2D Life-like CA (consolidated, Moore neighbourhood)
│   ├── game_of_life.py    # Standalone Pygame Game of Life (B3/S23)
│   ├── game_board.py      # Pygame board: draw cells, toggle, run simulation
│   ├── grapher.py         # Matplotlib visualiser: standard / detailed / heatmap modes
│   ├── drawing.py         # Pygame drawing board for creating initial states
│   ├── dataset_builder.py # Generates CSV dataset of (rule, initial, final) triples
│   ├── main_run.py        # Launches interactive 2D CA simulation
│   └── rule_scraper.py    # Scrapes Life-like rule table from conwaylife.com
└── notebooks/
    └── Random Forest.ipynb   # Rule prediction from CA snapshots (WIP)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the interactive 2D CA

```bash
cd packages/classical_ca/src
python main_run.py
```

- **Space** — start/pause evolution
- **Click + drag** — draw live cells

## Running Game of Life

```bash
python game_of_life.py
```

## 1D CA example

```python
from ca_1d import OneDimensionalCA

ca = OneDimensionalCA(rule_number=110, width=200, generations=100)
ca.initialize(1)   # single seed in centre
ca.evolve()
ca.visualize()
```

## Life-like rule notation

Rules follow Golly/RLE notation: `B<birth>/S<survival>`, e.g. `B3/S23` for Conway's Life.
Pass birth and survival as lists of digit strings to `TwoDimensionalCA`.
