---
title: Life-like cellular automata
page: lifelike
order: 2
---
A two-dimensional grid where a cell's next state depends on whether it is alive and how many of its eight neighbours are alive. A rule is written B/S: the neighbour counts that cause birth, and the counts that let a living cell survive. Conway's Life is B3/S23.

Try the named rules, place patterns such as gliders and the Gosper gun, or draw your own.

## Two mathematical facts

**Duality.** Swap every live and dead cell and a life-like rule turns into another life-like rule, its *dual*. Life, B3/S23, becomes B0123478/S01234678: the same game with live and dead swapped (and, because empty space is now "born" into, a flickering background). Day & Night is its own dual. Exactly 512 of the 262,144 rules are self-dual.

**Mean field.** Pretend every cell is independent with density ρ and you can write down the next density in closed form. Its fixed points predict where a random soup settles. For Life the prediction is a density of 0.370, with an unstable threshold at 0.1925: start below it and everything dies out in the model.

## Where the prediction fails

Simulating each rule on a 127 by 127 torus from a random start and averaging the last 50 of 400 steps:

| rule | mean-field density | simulated density |
|---|---|---|
| Seeds | 0.237 | 0.211 |
| Day & Night | 0.500 | 0.535 |
| Maze | 0.523 | 0.552 |
| Life | 0.370 | 0.062 |
| HighLife | 0.389 | 0.062 |
| Coral | 0.000 | 0.809 |
| Diamoeba | 0.000 | 0.926 |

Mean field works for gas-like rules such as Seeds, where cells really are close to independent. It fails where spatial structure matters: Life leaves behind sparse ash of still lifes and oscillators, so its true density is six times lower than predicted, while Coral and Diamoeba grow clusters that mean field cannot see. That gap is the motivation for simulating every rule.

## The atlas of all 131,072 rules

Rules that make empty space come alive (birth on zero neighbours) flash the whole grid every step, so the atlas covers the 131,072 rules without that property. Each one was run once on a 61 by 61 torus (a prime size, to avoid parity artefacts) for 200 steps, from the *same* random start, together with a copy that differs in a single cell to measure how damage spreads. For every rule we recorded the final density, how much changes per step, how far the damage spreads, how compressible the final picture is, Langton's λ and the mean-field prediction, then mapped the table with PCA and k-means.

Four groups appear. About 73% of the rules (95,355) behave like noise: about half the cells alive, about half of them changing each step, a one-cell difference spreading to roughly half the grid, and a final picture that does not compress at all. Another 13% (17,340) die out into sparse ash, and 14% (18,176) freeze into a dense, nearly static state. A tiny group of 201 rules (0.2%) is very active but not noisy: almost every cell changes each step, and the density swings widely, the signature of a grid that flashes.

## Is there an edge of chaos?

Langton proposed that interesting behaviour lives at intermediate values of λ, the fraction of neighbourhoods that map to a live cell. Mitchell, Hraber and Crutchfield later challenged how general that claim is. Binning the 131,072 rules into 20 equal-sized groups by λ, the median activity and damage start low (0.374 and 0.359 for the lowest λ), peak around λ = 0.5 (0.456 and 0.494) and fall again at the highest λ (0.415 and 0.396), while the median compression stays at the incompressible plateau of about 1.02 almost everywhere. The relationship is an inverted U, so a rank correlation understates it (Spearman 0.078 for activity and 0.077 for damage), whereas λ predicts the final density very well (0.85).

What the data do **not** show is a sharp transition to a rare, structured regime at intermediate λ: most random rules in the middle are simply noise. Structure is rare in rule space, and λ alone does not find it. This is one run per rule from one start, so it is an overview, not a classification of any individual rule.

