# Real-ESRGAN
Real-ESRGAN function for VapourSynth, based on https://github.com/xinntao/Real-ESRGAN.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchaudio` is not required and hence can be omitted from the command.
- [VapourSynth](http://www.vapoursynth.com/)


## Installation
```
pip install --upgrade vsrealesrgan
python -m vsrealesrgan
```


## Usage
```python
from vsrealesrgan import RealESRGAN

ret = RealESRGAN(clip)
```

See `__init__.py` for the description of the parameters.
