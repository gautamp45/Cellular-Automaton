---
title: Recovering the rule
page: inference
order: 5
---
Run a cellular automaton forward and you get a picture. Run the question backwards and you get inference: from observed states, which rule produced them? For life-like rules the answer can be searched exhaustively, estimated with Bayesian statistics, or learned by gradient descent through a differentiable simulator.

## Three ways to recover a rule

**Bayesian counting.** Show the system a grid and the grid one step later. Every cell tells you the output of one entry of the rule's lookup table: "a live cell with two live neighbours stayed alive". Count what each entry did, and in our tests a single 64 by 64 random pair pins down the rule exactly. With 5% of the outputs flipped, the posterior still picked every observed entry correctly in our tests, and it tells you how sure it is. Entries the data never exercised stay at 50/50: a lone glider simply does not contain enough situations to identify Life.

**Exhaustive search.** There are only 262,144 life-like rules, so you can simulate them all. For a 24 by 24 grid observed after two steps, simulating all of them takes a few seconds on a CPU, and in the reference case exactly one rule is consistent. When the search space is this small, brute force is exact and still affordable; machine learning earns its place when the space is too large to enumerate.

**A rule is a tiny neural network.** Count the live neighbours, add half the cell itself, and you have a number between 0 and 8.5. Any life-like rule is a function of that number on 18 possible values, and a function like that is exactly a small ReLU network. Conway's Life needs only four hidden neurons, with their kinks at 2, 2.5, 3.5 and 4. Across all 262,144 rules the number of neurons needed averages exactly 12.5.

## Can gradient descent do it instead?

Relax the lookup table into 18 real numbers, make the cellular automaton differentiable, and let gradient descent adjust the numbers until the simulated future matches the observed one. We tried this on 20 rules drawn evenly from the four groups of the rule-space atlas, on 32 by 32 grids, with 10 random restarts each, for rollouts of 1 to 4 steps and 0%, 2% and 5% noise on the observed output.

One step ahead, the problem is a sum of independent logistic losses, which is convex, and gradient descent finds the rule from every restart: 100% success without noise (95% interval 98 to 100%), 96.5% at 2% noise and 100% at 5%. Two steps ahead the composition makes the landscape non-convex, and the share of restarts that succeed falls to about a third (32%, interval 26 to 39%), then 19% at three steps and 8% at four. Restarts help a lot early on: at two steps, at least one of ten restarts succeeds on 90% of the instances, but only 50% at three steps and 20% at four. Noise barely matters; depth does.

Speed does not favour the exhaustive search either: ten restarts take 1 to 2 seconds, whereas simulating all 262,144 rules takes 8 to 20 seconds on these grids. The difference is reliability, not time: the exhaustive search is exact at every depth, and gradient descent is only dependable when the problem is convex.

About half of the 20 rules come from groups that die out or freeze, where most of the table is never exercised; the study scores bit errors only on entries the data can identify (on average 16 to 18 of the 18).

