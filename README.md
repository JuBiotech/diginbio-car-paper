# Installation of the Environment

The model was built with the development version of PyMC3 `v4.0.0`.
Why?
Because we can.

```bash
conda create -n pm3v4 -c conda-forge "python=3.8" libpython mkl-service m2w64-toolchain numba python-graphviz scipy
conda activate pm3v4
pip install git+https://github.com/pymc-devs/pymc3.git@main
```

