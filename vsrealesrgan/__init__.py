import numpy as np
import os
import torch
import vapoursynth as vs
from .rrdbnet_arch import RRDBNet
from .utils import RealESRGANer


def RealESRGAN(clip: vs.VideoNode, scale: int=2, anime: bool=False, tile: int=0, tile_pad: int=10, pre_pad: int=0,
               half: bool=False, device_type: str='cuda', device_index: int=0) -> vs.VideoNode:
    '''
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

    Parameters:
        clip: Clip to process. Only planar format with float sample type of 32 bit depth is supported.

        scale: Upsample scale factor of the network. Must be 2 or 4.

        anime: Use model optimized for anime. Currently only x4 is supported.

        tile: Tile size, 0 for no tile.

        tile_pad: Tile padding.

        pre_pad: Pre padding size at each border.

        half: Use half precision.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.
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

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    model_name = f'RealESRGAN_x{scale}plus' + ('_anime_6B' if anime else '') + '.pth'
    model_path = os.path.join(os.path.dirname(__file__), model_name)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6 if anime else 23, num_grow_ch=32, scale=scale)

    upsampler = RealESRGANer(device, scale, model_path, model, tile, tile_pad, pre_pad, half)

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)

    def realesrgan(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_tensor(f[0])
        output = upsampler.enhance(img)
        return tensor_to_frame(output, f[1])

    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f.get_read_array(plane)) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.data.squeeze().cpu().numpy()
    fout = f.copy()
    for plane in range(fout.format.num_planes):
        np.copyto(np.asarray(fout.get_write_array(plane)), arr[plane, :, :])
    return fout
