# GraphOpt

Implementation of a graph optimization solver similar to G2O for solving nonlinear least squares problems.  

## Dependencies

The optimizer is tested on Ubuntu 20.04 with Eigen 3.3.

## Build

```bash
git clone https://github.com/Kindn/GraphOpt.git
cd GraphOpt
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=1
make
```

## Run Examples

* Curve fitting

```bash
# In the build directory
./example/curve_fitting/curve_fitting
```

* BAL Dataset

```bash
# In the build directory
./example/bundle_adjustment/bundle_adjustment /path/to/dataset.txt
```

