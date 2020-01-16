# To add CG method to PETSc

First, implement the method in `src/ksp/ksp/impls/cg`

Then update:

- `src/ksp/ksp/impls/cg/makefile`
- `src/ksp/ksp/interface/itregis.c`
- `include/petscksp.h`

