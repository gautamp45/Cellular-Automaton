---
title: Neural cellular automata
page: nca
order: 4
---
In a neural cellular automaton every cell carries 16 numbers and runs the same tiny neural network on a fixed 3x3 view of its neighbours. The network is trained so that a single seed cell grows into a target image and repairs itself after damage.

Choose a trained model, change how often cells fire, and click the grid to damage it.

## How the models were evaluated

Every one of the 25 trained models was grown from a single seed cell for 1000 steps, 30 times, with a different random firing pattern each time. At step 300 a random disc of cells is erased, and the model has to regrow it. The damage is the same for a given seed across all models, so comparisons between models are paired: they differ only in the model, never in the luck of the draw.

For each run we record the error against the target image (mean squared error over RGBA) after 200 and 1000 steps, how much the error drifts, the worst error after the damage, and how many steps it takes to recover. Recovery is only measured for models that had converged to the target before the damage; for the others a relative threshold would report a meaningless "recovered in one step". Confidence intervals are 95% percentile bootstraps over the 30 seeds, and differences between configurations use a paired sign-flip permutation test with Holm correction.

**What these intervals do and do not cover.** They capture inference randomness only: which cells fire and where the damage lands. Each configuration was *trained once*, so the variation between training runs is not measured, and an apparent effect of a filter or a loss may partly be training luck.

## What we found

**Which filter?** Models that perceive their neighbours through direction-aware filters, Sobel or Scharr, converge to an error of about 0.0002 to 0.0006 on all three targets. Models given symmetric filters (Gaussian, Laplacian or mean) cannot tell which way is which and stay 20 to 600 times worse (error 0.009 to 0.047 after 200 steps, never below 0.005). Sobel versus Scharr is a statistical but not a practical difference: it is below 0.0001 in absolute error and its sign depends on the target. Every comparison above is paired by seed and Holm-corrected; the permutation p-values are all below 0.0001, the smallest value 100,000 permutations can resolve.

**Which loss?** With Sobel perception, L1, L2 and Manhattan all reach the same ballpark (0.00015 to 0.0006 depending on target), while the Hinge loss never gets below 0.034, because it ignores errors smaller than 0.5 and so has no gradient to finish the job.

**Does training with damage help?** Yes, clearly. After the same damage, the model trained with damage regrows to its pre-damage error in about 79 steps (95% interval 71 to 88), against about 193 (160 to 232) for the same model trained without damage, and its worst error right after the damage is lower. Several converged models do not always recover within the 700 steps after the damage: for example the chest models trained with L1 or Manhattan loss recovered in 18 and 17 of 30 runs.

**A warning about the huge effect sizes.** The paired effect sizes are enormous (a Cohen's d of several hundred for isotropic against Sobel filters), not because the effects are that large but because the run-to-run variation being measured is tiny. A fixed model grown again from the same seed behaves almost identically each time. The variation that matters, between models trained again from scratch, is not in these numbers.

