import os

import numpy as np
import torch
import vapoursynth as vs

from .rrdbnet_arch import RRDBNet
from .srvgg_arch import SRVGGNetCompact
from .utils import RealESRGANer

dirname = os.path.dirname(__file__)


def RealESRGAN(
    clip: vs.VideoNode,
    model_type: int = 0,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 10,
    device_type: str = 'cuda',
    device_index: int = 0,
    fp16: bool = False,
) -> vs.VideoNode:
    '''
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        model_type: Model type to use.
            0 = RealESRGAN_x2plus (x2 model for general images)
            1 = RealESRGAN_x4plus (x4 model for general images)
            2 = RealESRGAN_x4plus_anime_6B (x4 model optimized for anime images)
            3 = RealESRGANv2-animevideo-xsx2 (x2 model optimized for anime videos)
            4 = RealESRGANv2-animevideo-xsx4 (x4 model optimized for anime videos)

        tile_w, tile_h: Tile width and height respectively, 0 for no tiling.
            It's recommended that the input's width and height is divisible by the tile's width and height respectively.
            Set it to the maximum value that your GPU supports to reduce its impact on the output.

        tile_pad: Tile padding.

        device_type: Device type on which the tensor is allocated. Must be 'cuda' or 'cpu'.

        device_index: Device ordinal for the device type.

        fp16: fp16 mode for faster and more lightweight inference on cards with Tensor Cores.
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if model_type < 0 or model_type > 4:
        raise vs.Error('RealESRGAN: model_type must be 0, 1, 2, 3, or 4')

    device_type = device_type.lower()

    if device_type not in ['cuda', 'cpu']:
        raise vs.Error("RealESRGAN: device_type must be 'cuda' or 'cpu'")

    if device_type == 'cuda' and not torch.cuda.is_available():
        raise vs.Error('RealESRGAN: CUDA is not available')

    if os.path.getsize(os.path.join(dirname, 'RealESRGAN_x2plus.pth')) == 0:
        raise vs.Error("RealESRGAN: model files have not been downloaded. run 'python -m vsrealesrgan' first")

    device = torch.device(device_type, device_index)
    if device_type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if model_type == 0:  # x2 RRDBNet model
        model_name = 'RealESRGAN_x2plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif model_type == 1:  # x4 RRDBNet model
        model_name = 'RealESRGAN_x4plus.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_type == 2:  # x4 RRDBNet model with 6 blocks
        model_name = 'RealESRGAN_x4plus_anime_6B.pth'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif model_type == 3:  # x2 VGG-style model (XS size)
        model_name = 'RealESRGANv2-animevideo-xsx2.pth'
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    else:  # x4 VGG-style model (XS size)
        model_name = 'RealESRGANv2-animevideo-xsx4.pth'
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    model_path = os.path.join(dirname, model_name)

    upsampler = RealESRGANer(
        device=device, scale=netscale, model_path=model_path, model=model, tile_x=tile_w, tile_y=tile_h, tile_pad=tile_pad, pre_pad=0, half=fp16
    )

    @torch.inference_mode()
    def realesrgan(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_tensor(f[0])
        output = upsampler.enhance(img)
        return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * netscale, height=clip.height * netscale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_tensor(f: vs.VideoFrame) -> torch.Tensor:
    arr = np.stack([np.asarray(f[plane]) for plane in range(f.format.num_planes)])
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_frame(t: torch.Tensor, f: vs.VideoFrame) -> vs.VideoFrame:
    arr = t.squeeze(0).detach().cpu().numpy()
    for plane in range(f.format.num_planes):
        np.copyto(np.asarray(f[plane]), arr[plane, :, :])
    return f
