# Installation of the Environment

The model was built with the development version of PyMC3 `v4.0.0`.
Why?
Because we can.

```bash
conda create -n CARenv -c conda-forge "python=3.8" libpython mkl-service m2w64-toolchain numba python-graphviz scipy jupyter openpyxyl
conda activate CARenv
pip install git+https://github.com/pymc-devs/pymc3.git@main
pip install calibr8 robotools
```

