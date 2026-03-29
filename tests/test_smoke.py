"""Smoke tests for MinkowskiEngine installation."""
import pytest
import torch


def test_import():
    import MinkowskiEngine as ME
    assert hasattr(ME, '__version__')
    assert ME.__version__ == '0.6.0'


def test_sparse_tensor_cpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 3)
    x = ME.SparseTensor(feats, coords)
    assert x.shape == torch.Size([100, 3])


def test_convolution_cpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 3)
    x = ME.SparseTensor(feats, coords)
    conv = ME.MinkowskiConvolution(3, 16, kernel_size=3, stride=1, dimension=3)
    out = conv(x)
    assert out.shape[1] == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_tensor_gpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 3).cuda()
    x = ME.SparseTensor(feats, coords, device='cuda')
    assert x.F.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_convolution_gpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 3).cuda()
    x = ME.SparseTensor(feats, coords, device='cuda')
    conv = ME.MinkowskiConvolution(3, 16, kernel_size=3, stride=1, dimension=3).cuda()
    out = conv(x)
    assert out.shape[1] == 16
    assert out.F.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pooling_gpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 3).cuda()
    x = ME.SparseTensor(feats, coords, device='cuda')
    pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3).cuda()
    out = pool(x)
    assert out.F.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batchnorm_gpu():
    import MinkowskiEngine as ME
    coords = torch.randint(0, 100, (100, 4)).int()
    feats = torch.randn(100, 16).cuda()
    x = ME.SparseTensor(feats, coords, device='cuda')
    bn = ME.MinkowskiBatchNorm(16).cuda()
    out = bn(x)
    assert out.shape == x.shape
