# About

# Usage 

## tree_classification.py

1. Generate CHM from point cloud
    ```
    python3 python3 generate_chm_from_laz.py <las/laz file name>
    ```
    **NOTE**: Use double backslashes `\\` to separate the directory and file names. E.g., `dir1\\dir2\\file.laz` 

    A generated CHM image is saved in `CHM/`.

1. Run `tree_classification.py`
    ```
     python3 tree_classification.py <file1> <file2> --root <file root>
    ```

# Requirements

## Environment

The software is tested `Ubuntu 20.04` on `WSL2` on `Windows 11`.

## Software

### LAStools

- Install [LAStools](https://rapidlasso.com/lastools/) on the host (Windows) environment.
- Add the path to `LAStools/bin` to `PATH` in WSL.

## Python libraries

```
pip3 install -r requirements.txt
```

### Basics
- `numpy`
- `matplotlib`
### Point cloud analysis
- `laspy` (NOT `pylas`)
- `open3d`

### Visualization
- `glooey`
- `pyglet`
- `tqdm`

### A library that you need source build
- `semantic-octomap-python`
  ```
  cd thirdparty/octomap-python/.
  python setup.py build_ext
  ```

# TODO
