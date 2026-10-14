---
title: Elementary cellular automata
page: elementary
order: 1
---
A row of cells, each 0 or 1. The next state of a cell depends on itself and its two neighbours, so a rule is a lookup table with eight entries, and there are exactly 256 rules. Rule 30 looks random, rule 90 draws the Sierpinski triangle, and rule 110 is Turing-complete.

Pick a rule, toggle its eight bits, and watch the space-time diagram unfold.

## Things worth knowing

There are 256 rules, but only 88 genuinely different ones: swapping left and right, and swapping 0 and 1, turn a rule into an equivalent one (rule 30 shares its class with 86, 135 and 149).

Rule 90 is *surjective* on the infinite line, meaning every pattern has a predecessor. On a ring of 12 cells, though, 3072 of the 4096 configurations have no predecessor at all. Surjectivity on the infinite line does not carry over to finite rings.

Rule 90 on a ring whose size is a power of two erases everything. Real experiments use prime widths such as 251 to avoid this.

The behaviour atlas measures how fast damage spreads, how compressible the pattern is, how active it is, and how it repeats, then groups the 256 rules without using Wolfram's classes. The groups recover a coarse version of those classes; the boundary between complex and chaotic rules stays fuzzy, as in the literature.

