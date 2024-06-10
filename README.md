# Real-ESRGAN
Training Real-World Blind Super-Resolution with Pure Synthetic Data, based on https://github.com/xinntao/Real-ESRGAN.


## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.4.0.dev or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later

`trt` requires additional Python packages:
- [TensorRT](https://developer.nvidia.com/tensorrt/) 10.0.1
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.4.0.dev

To install TensorRT, run `pip install tensorrt==10.0.1 tensorrt-cu12_bindings==10.0.1 tensorrt-cu12_libs==10.0.1 --extra-index-url https://pypi.nvidia.com`

To install Torch-TensorRT, Windows users can pip install the whl file on [Releases](https://github.com/HolyWu/vs-realesrgan/releases). Linux users can run `pip install --pre torch_tensorrt --index-url https://download.pytorch.org/whl/nightly/cu124` (requires PyTorch nightly build).


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
