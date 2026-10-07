# P00: Repo hygiene, archive, removals

**Tier:** Must · **Prereqs:** none · **Touches no runtime code.**

## Steps

1. Remove tracked junk:
   ```bash
   git rm -r --cached .DS_Store .vscode "CA_Cryptography System/__pycache__" "Rule Predictor/__pycache__" "TD CA/__pycache__" __pycache__
   rm -rf .DS_Store .vscode "CA_Cryptography System/__pycache__" "Rule Predictor/__pycache__" "TD CA/__pycache__" __pycache__
   git rm "TD CA/UI.py" neural-cellular-automata/.gitignore
   ```
2. Root `.gitignore`:
   ```gitignore
   __pycache__/
   *.py[cod]
   *.egg-info/
   .venv/
   venv/
   build/
   dist/
   .pytest_cache/
   .ruff_cache/
   .mypy_cache/
   .hypothesis/
   logs/
   runs/
   .DS_Store
   .vscode/
   .idea/
   node_modules/
   web/dist/
   ```
3. `.gitattributes`:
   ```gitattributes
   * text=auto
   *.pt binary
   *.npz binary
   *.bin binary
   *.gif binary
   *.png binary
   ```
4. Archive crypto: `git mv "CA_Cryptography System" experiments/ca_crypto`. Do not edit its code. Add `experiments/ca_crypto/README.md` containing:
   - What it is: a 1D CA as a keystream generator. `main.py` is the XOR-keystream variant, `Crypt_CA_1.py` the forward-evolution variant, `Attacks.py` a preimage brute force.
   - How to run: `cd experiments/ca_crypto && python main.py` (needs matplotlib, opencv-python).
   - Credits: Manan Shah (attack and visualiser commit).
   - Known issues B8, B9, B10 from `docs/plan/AUDIT.md`.
   - An honest assessment: with a random initial state as long as the message, the scheme is a one-time pad with extra steps. The rule (8 bits) plus generations (~10 bits) add at most ~18 bits of key. Also, `Crypt_CA_1` cannot decrypt because only 6 of 256 elementary rules are reversible. P09's preimage counter is the principled version of `Attacks.py`.
5. `git rm -r "Rule Predictor"` (decision D2; rebuilt properly in P11).
6. `git rm neural-cellular-automata/notebooks/Growing_Neural_Cellular_Automata.ipynb` (D4; README will link the original Colab).

## Done when
- `git ls-files | grep -E "pycache|DS_Store|vscode|Rule Predictor"` prints nothing.
- `experiments/ca_crypto/README.md` exists.

**Commit:** `chore: remove committed caches, archive crypto experiment, drop broken rule predictor`
