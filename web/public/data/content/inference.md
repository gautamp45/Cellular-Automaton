---
title: Recovering the rule
page: inference
order: 5
---
Run a cellular automaton forward and you get a picture. Run the question backwards and you get inference: from observed states, which rule produced them? For life-like rules the answer can be searched exhaustively, estimated with Bayesian statistics, or learned by gradient descent through a differentiable simulator.

## Three ways to recover a rule

**Bayesian counting.** Show the system a grid and the grid one step later. Every cell tells you the output of one entry of the rule's lookup table: "a live cell with two live neighbours stayed alive". Count what each entry did, and in our tests a single 64 by 64 random pair pins down the rule exactly. With 5% of the outputs flipped, the posterior still picked every observed entry correctly in our tests, and it tells you how sure it is. Entries the data never exercised stay at 50/50: a lone glider simply does not contain enough situations to identify Life.

**Exhaustive search.** There are only 262,144 life-like rules, so you can simulate them all. For a 24 by 24 grid observed after two steps, simulating all of them takes a few seconds on a CPU, and in the reference case exactly one rule is consistent. When the search space is this small, brute force is cheaper and exact; machine learning earns its place when the space is too large to enumerate.

**A rule is a tiny neural network.** Count the live neighbours, add half the cell itself, and you have a number between 0 and 8.5. Any life-like rule is a function of that number on 18 possible values, and a function like that is exactly a small ReLU network. Conway's Life needs only four hidden neurons, with their kinks at 2, 2.5, 3.5 and 4. Across all 262,144 rules the number of neurons needed averages exactly 12.5.

