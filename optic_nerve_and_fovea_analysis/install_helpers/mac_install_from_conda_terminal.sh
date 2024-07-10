#!/bin/sh

# by the end of October 2023, these next two lines regarding libmamba should be unnecessary... https://conda.org/blog/2023-07-05-conda-libmamba-solver-rollout/
# until then, these lines regarding libmamba are needed to reduce install time from an hour to a minute
conda install -y -n base conda-libmamba-solver
conda config --set solver libmamba

# after October 2023 (and after you update or reinstall anaconda), this script should be able to just start here
conda install -y -c conda-forge \
  "pyqt>=5.15,<6" \
  "napari>=0.4.18,<0.5" \
  "imagecodecs>=2021.8,<2024" \
  "numba>=0.56,<0.58" \
  "numpy>=1.21,<1.26" \
  "scikit-image>=0.19,<0.22" \
  "shapely>=2.0,<3.0" \
  "onnxruntime>=1.14.0,<2.0"
