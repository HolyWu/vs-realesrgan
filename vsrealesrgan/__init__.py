import os

import numpy as np
import torch
import vapoursynth as vs

from .rrdbnet_arch import RRDBNet
from .utils import RealESRGANer

vs_api_below4 = vs.__api_version__.api_major < 4


def RealESRGAN(clip: vs.VideoNode, scale: int = 2, anime: bool = False, tile_x: int = 0, tile_y: int = 0, tile_pad: int = 10, pre_pad: int = 0,
               device_type: str = 'cuda', device_index: int = 0, fp16: bool = False) -> vs.VideoNode:
    '''
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        scale: Upsample scale factor of the network. Must be 2 or 4.

        anime: Use model optimized for anime. Currently only x4 is supported.

        tile_x, tile_y: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        pre_pad: Pre padding size at each border.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if scale not in [2, 4]:
        raise vs.Error('RealESRGAN: scale must be 2 or 4')

    if anime and scale == 2:
        raise vs.Error('RealESRGAN: only x4 is supported for anime model')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("RealESRGAN: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('RealESRGAN: CUDA is not available')

    if os.path.getsize(os.path.join(os.path.dirname(__file__), 'RealESRGAN_x2plus.pth')) == 0:
        raise vs.Error("RealESRGAN: model files have not been downloaded. run 'python -m vsrealesrgan' first")

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model_name = f'RealESRGAN_x{scale}plus' + ('_anime_6B' if anime else '') + '.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6 if anime else 23, num_grow_ch=32, scale=scale)

    upsampler = RealESRGANer(device, scale, model_path, model, tile_x, tile_y, tile_pad, pre_pad, fp16)

    @torch.inference_mode()
    def realesrgan(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_tensor(f[0])
        output = upsampler.enhance(img)
        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane) if vs_api_below4 else f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f.get_write_array(plane) if vs_api_below4 else f[plane]), arr[plane, :, :])
    return f
