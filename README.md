[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/

# MinkowskiEngine-new

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.0-ee4c2c)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

A modernized fork of [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) maintained by [ilessiorobotflowlabs](https://github.com/ilessiorobotflowlabs). Full credit to the original authors -- Christopher Choy and NVIDIA -- for the foundational work on sparse tensor auto-differentiation.

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

---

## What's New in v0.6.0

- **PyTorch 2.x support** -- tested with PyTorch 2.5+ and 2.10+
- **CUDA 12.x support**
- **Python 3.10-3.12 support**
- **Modern `pyproject.toml` build system** -- replaces legacy `setup.cfg` / `numpy.distutils`
- **`uv` package manager support** -- fast, reliable installs
- **Removed deprecated APIs** -- `numpy.distutils`, `DeprecatedTypeProperties.h`, and other legacy code paths
- **Fixed cuSPARSE handle leak** -- replaced `at::cuda::getCurrentCUDASparseHandle` with a proper handle manager
- **Proper relative imports throughout** -- eliminated import fragility across the package

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA 12.x (for GPU builds)
- GCC >= 7.4.0
- ninja (for compilation)

### Install with uv (recommended)

```bash
# Install directly from GitHub
uv pip install git+https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new.git

# Or clone and install in editable mode
git clone https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new.git
cd MinkowskiEngine-new
uv pip install -e .

# CPU-only build
CPU_ONLY=1 uv pip install -e .

# Force CUDA build
FORCE_CUDA=1 uv pip install -e .
```

### Install with pip

```bash
pip install git+https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new.git

# Or clone and install
git clone https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new.git
cd MinkowskiEngine-new
pip install -e .
```

### Build options

You can control the build with environment variables:

```bash
# Specify CUDA home directory
export CUDA_HOME=/usr/local/cuda-12

# Specify CUDA architectures
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Force CUDA or CPU-only build
FORCE_CUDA=1 uv pip install -e .
CPU_ONLY=1 uv pip install -e .
```

### Docker

```bash
git clone https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new.git
cd MinkowskiEngine-new
docker build -t minkowski_engine docker
docker run minkowski_engine python3 -c "import MinkowskiEngine; print(MinkowskiEngine.__version__)"
```

---

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://nvidia.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.classification_modelnet40`                                                          |
| Reconstruction        | <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://nvidia.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                       |
| Detection             | <img src="https://nvidia.github.io/MinkowskiEngine/_images/detection_3d_net.png">                                                                                               |


## Sparse Tensor Networks: Neural Networks for Spatially Sparse Tensors

Compressing a neural network to speedup inference and minimize memory footprint has been studied widely. One of the popular techniques for model compression is pruning the weights in convnets, is also known as [*sparse convolutional networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf). Such parameter-space sparsity used for model compression compresses networks that operate on dense tensors and all intermediate activations of these networks are also dense tensors.

However, in this work, we focus on [*spatially* sparse data](https://arxiv.org/abs/1409.6070), in particular, spatially sparse high-dimensional inputs and 3D data and convolution on the surface of 3D objects, first proposed in [Siggraph'17](https://wang-ps.github.io/O-CNN.html). We can also represent these data as sparse tensors, and these sparse tensors are commonplace in high-dimensional problems such as 3D perception, registration, and statistical data. We define neural networks specialized for these inputs as *sparse tensor networks*  and these sparse tensor networks process and generate sparse tensors as outputs. To construct a sparse tensor network, we build all standard neural network layers such as MLPs, non-linearities, convolution, normalizations, pooling operations as the same way we define them on a dense tensor and implemented in the Minkowski Engine.

We visualized a sparse tensor network operation on a sparse tensor, convolution, below. The convolution layer on a sparse tensor works similarly to that on a dense tensor. However, on a sparse tensor, we compute convolution outputs on a few specified points which we can control in the [generalized convolution](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html). For more information, please visit [the documentation page on sparse tensor networks](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html) and [the terminology page](https://nvidia.github.io/MinkowskiEngine/terminology.html).

| Dense Tensor                                                                | Sparse Tensor                                                                |
|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_dense.gif"> | <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_sparse.gif"> |

--------------------------------------------------------------------------------

## Features

- Unlimited high-dimensional sparse tensor support
- All standard neural network layers (Convolution, Pooling, Broadcast, etc.)
- Dynamic computation graph
- Custom kernel shapes
- Multi-GPU training
- Multi-threaded kernel map
- Multi-threaded compilation
- Highly-optimized GPU kernels


## Quick Start

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`MinkowskiEngine.utils.sparse_quantize`).


### Creating a Network

```python
import torch.nn as nn
import MinkowskiEngine as ME

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)
```

### Forward and backward using the custom network

```python
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    input = ME.SparseTensor(feat, coordinates=coords)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)
```

## Discussion and Documentation

For discussion and questions, please use `minkowskiengine@googlegroups.com`.
For API and general usage, please refer to the [MinkowskiEngine documentation
page](http://nvidia.github.io/MinkowskiEngine/) for more detail.

For issues not listed on the API and feature requests, feel free to submit
an issue on the [github issue
page](https://github.com/ilessiorobotflowlabs/MinkowskiEngine-new/issues).


## Known Issues

### Specifying CUDA architecture list

In some cases, you need to explicitly specify which compute capability your GPU uses. The default list might not contain your architecture.

```bash
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
FORCE_CUDA=1 uv pip install -e .
```

### Too much GPU memory usage or Frequent Out of Memory

There are a few causes for this error.

1. Out of memory during a long running training

MinkowskiEngine is a specialized library that can handle different number of points or different number of non-zero elements at every iteration during training, which is common in point cloud data.
However, pytorch is implemented assuming that the number of point, or size of the activations do not change at every iteration. Thus, the GPU memory caching used by pytorch can result in unnecessarily large memory consumption.

Specifically, pytorch caches chunks of memory spaces to speed up allocation used in every tensor creation. If it fails to find the memory space, it splits an existing cached memory or allocate new space if there's no cached memory large enough for the requested size. Thus, every time we use different number of point (number of non-zero elements) with pytorch, it either split existing cache or reserve new memory. If the cache is too fragmented and allocated all GPU space, it will raise out of memory error.

**To prevent this, you must clear the cache at regular interval with `torch.cuda.empty_cache()`.**

### Running the MinkowskiEngine on nodes with a large number of CPUs

The MinkowskiEngine uses OpenMP to parallelize the kernel map generation. However, when the number of threads used for parallelization is too large (e.g. OMP_NUM_THREADS=80), the efficiency drops rapidly as all threads simply wait for multithread locks to be released.
In such cases, set the number of threads used for OpenMP. Usually, any number below 24 would be fine, but search for the optimal setup on your system.

```
export OMP_NUM_THREADS=<number of threads to use>; python <your_program.py>
```

## Citing Minkowski Engine

If you use the Minkowski Engine, please cite:

- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755), [[pdf]](https://arxiv.org/pdf/1904.08755.pdf)

```
@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}
```

For multi-threaded kernel map generation, please cite:

```
@inproceedings{choy2019fully,
  title={Fully Convolutional Geometric Features},
  author={Choy, Christopher and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8958--8966},
  year={2019}
}
```

For strided pooling layers for high-dimensional convolutions, please cite:

```
@inproceedings{choy2020high,
  title={High-dimensional Convolutional Networks for Geometric Pattern Recognition},
  author={Choy, Christopher and Lee, Junha and Ranftl, Rene and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

For generative transposed convolution, please cite:

```
@inproceedings{gwak2020gsdn,
  title={Generative Sparse Detection Networks for 3D Single-shot Object Detection},
  author={Gwak, JunYoung and Choy, Christopher B and Savarese, Silvio},
  booktitle={European conference on computer vision},
  year={2020}
}
```


## Projects using Minkowski Engine

Please feel free to update [the wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage) to add your projects!

- [Projects using MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage)

- Segmentation: [3D and 4D Spatio-Temporal Semantic Segmentation, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- Representation Learning: [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- 3D Registration: [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- 3D Registration: [Deep Global Registration, CVPR'20](https://arxiv.org/abs/2004.11540)
- Pattern Recognition: [High-Dimensional Convolutional Networks for Geometric Pattern Recognition, CVPR'20](https://arxiv.org/abs/2005.08144)
- Detection: [Generative Sparse Detection Networks for 3D Single-shot Object Detection, ECCV'20](https://arxiv.org/abs/2006.12356)
- Image matching: [Sparse Neighbourhood Consensus Networks, ECCV'20](https://www.di.ens.fr/willow/research/sparse-ncnet/)
