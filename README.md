# Real-ESRGAN
Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

Ported from https://github.com/xinntao/Real-ESRGAN


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started), preferably with CUDA. Note that `torchvision` and `torchaudio` are not required and hence can be omitted from the command.
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
