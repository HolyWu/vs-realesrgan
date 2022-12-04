# Real-ESRGAN
Training Real-World Blind Super-Resolution with Pure Synthetic Data, based on https://github.com/xinntao/Real-ESRGAN.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13.0+
- [VapourSynth](http://www.vapoursynth.com/) R55+

`trt` requires additional runtime libraries:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn)
- [TensorRT](https://developer.nvidia.com/tensorrt)

For ease of installation on Windows, you can download the 7z file on [Releases](https://github.com/HolyWu/vs-rife/releases) which contains required runtime libraries and Python wheel file. Either add the unzipped directory to your system `PATH` or copy the DLL files to a directory which is already in your system `PATH`. Finally pip install the Python wheel file.


## Installation
```
pip install -U vsrealesrgan
python -m vsrealesrgan
```


## Usage
```python
from vsrealesrgan import RealESRGAN

ret = RealESRGAN(clip)
```

See `__init__.py` for the description of the parameters.
