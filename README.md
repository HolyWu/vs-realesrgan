# Real-ESRGAN
Training Real-World Blind Super-Resolution with Pure Synthetic Data, based on https://github.com/xinntao/Real-ESRGAN.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0.dev
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional packages:
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0.dev

To install the latest nightly build of PyTorch and Torch-TensorRT, run:
```
pip install -U packaging setuptools wheel
pip install --pre -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124 --extra-index-url https://pypi.nvidia.com
```


## Installation
```
pip install -U vsrealesrgan
python -m vsrealesrgan
```


## Usage
```python
from vsrealesrgan import realesrgan

ret = realesrgan(clip)
```

See `__init__.py` for the description of the parameters.
