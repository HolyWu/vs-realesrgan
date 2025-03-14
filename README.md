# Real-ESRGAN
Training Real-World Blind Super-Resolution with Pure Synthetic Data, based on https://github.com/xinntao/Real-ESRGAN.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0.dev or later

To install the latest stable version of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsrealesrgan
```

If you want to download all models at once, run `python -m vsrealesrgan`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `realesrgan()`.


## Usage
```python
from vsrealesrgan import realesrgan

ret = realesrgan(clip)
```

See `__init__.py` for the description of the parameters.
