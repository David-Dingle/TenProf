# torch-monitor

[![CodeFactor](https://www.codefactor.io/repository/github/jokeren/torch-monitor/badge?s=80fcc39bbf4c5c080ee7587b962fc5a6925ee67e)](https://www.codefactor.io/repository/github/jokeren/torch-monitor)

A Python and PyTorch monitoring interface

## Installation

#### 1 Install dependencies

In order to build you'll need the following packages:

-   cmake (>= 3.12)
-   pytorch
-   torchvision

#### 2 Build

Use the following commands to get source code and build torch-monitor:

```console
git clone https://github.com/Jokeren/torch-monitor.git
```

```console
mkdir build && cd build
```

```console
cmake .. -DTORCH_DIR=/path/to/torch/install/dir
```

```console
make
```

```console
make install
```
