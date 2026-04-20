# ca_cryptography

A 1D Cellular Automaton stream cipher. A CA evolves a random initial state for many
generations; the final state is used as a one-time-pad key to XOR-encrypt plaintext.

## Structure

```
ca_cryptography/
└── src/
    ├── ca.py           # 1D CA: initialise from a binary string, evolve, return final state
    ├── rsrg.py         # Cryptographically-random state / rule / generation generator
    ├── converter.py    # Text ↔ binary string conversion
    ├── enc_dec.py      # XOR encrypt / decrypt
    ├── visualiser.py   # 6-panel matplotlib visualisation (with decrypt panel)
    ├── visualizer_2.py # 5-panel matplotlib visualisation (encrypt only)
    ├── main.py         # Full encrypt + decrypt demo
    ├── crypt_ca_1.py   # Alternate demo: use plaintext as seed (no XOR)
    └── attacks.py      # Brute-force reverse-evolution attack demo
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd packages/ca_cryptography/src
python main.py
```

## How it works

1. `RandomStateRuleGen` generates a random initial state, rule, and generation count.
2. `OneDimensionalCA` evolves the initial state for *n* generations.
3. The final CA state is XOR-ed with the binary encoding of the plaintext.
4. Decryption is identical (XOR is its own inverse).
