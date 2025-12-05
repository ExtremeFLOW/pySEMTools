# Calculus in SEM mesh

In these example we show how one can perform differentiation and integration in the sem mesh.

Furthermore, we show how to reduce the discontinouties among elements. Note that this operation will work on elements that have neighbours in the same MPI rank. Searching for neighbours in other ranks is not implemented.

1. Derivation (Example 1.5 uses torch)
2. Integration