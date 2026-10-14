# ca_crypto: archived experiment

A 1D elementary cellular automaton used as a keystream generator. Archived as-is from the original
`CA_Cryptography System/` directory. The code is not linted, tested or packaged, and is not to be extended.

| File | What it does |
|---|---|
| `main.py` | XOR-keystream variant: evolve a random initial state, XOR the final state with the binary-encoded plaintext |
| `Crypt_CA_1.py` | Forward-evolution variant: evolves the plaintext itself |
| `Attacks.py` | Preimage brute force against the first variant |

## Run

```bash
cd experiments/ca_crypto && python main.py
```

Needs `matplotlib` and `opencv-python`, which are not dependencies of `cellauto`.

## Credits

Manan Shah (attack and visualiser commit).

## Known issues

These are numbered as in `docs/plan/AUDIT.md`.

- **B8**: `RSRG.py` uses `secrets.SystemRandom(seed)`, which ignores the seed. `randint(0, 256)` can return rule 256, and `randint(0, 1001)` can return 0 generations, after which `grid[-1]` raises `IndexError`.
- **B9**: `Crypt_CA_1.py` evolves the plaintext forward. Most rules are non-invertible (only 6 of the 256 elementary rules are reversible), so the ciphertext cannot be decrypted.
- **B10**: the brute-force loop in `Attacks.py` `break`s after rule 1, `recur_brute_force` is a stub, and importing the module runs `Crypt_CA_1`.

## Assessment

With a random initial state as long as the message, the scheme is a one-time pad with extra steps: the
security comes from the random initial state, not from the CA. If the state were short enough to be a
real key, the CA would add only the rule (8 bits) plus the number of generations (about 10 bits), so at
most ~18 bits of key, which is trivially searchable. And `Crypt_CA_1` cannot decrypt at all (B9).

The principled version of `Attacks.py` is the preimage counter in P09, which uses the transfer-matrix
method to count the preimages of a configuration exactly.
