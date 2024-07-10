call conda install -y -n base conda-libmamba-solver & ^
  conda config --set solver libmamba & ^
  conda install -y -c conda-forge ^
    pyqt~=2.3.1 ^
    napari~=0.4.18 ^
    imagecodecs>=2021.8,<=2023 ^
    numba>=0.56,<=0.57 ^
    numpy>=1.21,<=1.25 ^
    scikit-image>=0.19,<=0.21 ^
    onnxruntime>=1.14.0,<2.0
