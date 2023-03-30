from __future__ import annotations

import math
import os
from dataclasses import dataclass
from threading import Lock

import numpy as np
import tensorrt
import torch
import torch.nn.functional as F
import vapoursynth as vs
from functorch.compile import memory_efficient_fusion
from torch_tensorrt.fx import LowerSetting
from torch_tensorrt.fx.lower import Lowerer
from torch_tensorrt.fx.utils import LowerPrecision

from .rrdbnet_arch import RRDBNet
from .srvgg_arch import SRVGGNetCompact

__version__ = "4.0.1"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class CUDAGraphs:
        graph: list[torch.cuda.CUDAGraph]
        static_input: list[torch.Tensor]
        static_output: list[torch.Tensor]

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


@torch.inference_mode()
def realesrgan(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    nvfuser: bool = False,
    cuda_graphs: bool = False,
    trt: bool = False,
    trt_max_workspace_size: int = 1 << 30,
    trt_cache_path: str = model_dir,
    model: int = 4,
    model_path: str | None = None,
    denoise_strength: float = 0.5,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 8,
) -> vs.VideoNode:
    """Training Real-World Blind Super-Resolution with Pure Synthetic Data

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported.
                                    RGBH performs inference in FP16 mode while RGBS performs inference in FP32 mode.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param nvfuser:                 Enable fusion through nvFuser. Not allowed in TensorRT. (experimental)
    :param cuda_graphs:             Use CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
                                    sequentially. Not allowed in TensorRT.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_max_workspace_size:  Maximum workspace size for TensorRT engine.
    :param trt_cache_path:          Path for TensorRT engine file. Engine will be cached when it's built for the first
                                    time. Note each engine is created for specific settings such as model path/name,
                                    precision, workspace etc, and specific GPUs and it's not portable.
    :param model:                   Model to use.
                                    0 = ESRGAN_SRx4_DF2KOST_official-ff704c30 (official ESRGAN x4 model)
                                    1 = RealESRGAN_x2plus (x2 model for general images)
                                    2 = RealESRGAN_x4plus (x4 model for general images)
                                    3 = RealESRGAN_x4plus_anime_6B (x4 model optimized for anime images)
                                    4 = realesr-animevideov3 (x4 model optimized for anime videos)
                                    5 = realesr-general-x4v3 (tiny small x4 model for general scenes)
    :param model_path:              Path to custom model file.
    :param denoise_strength:        Denoise strength for realesr-general-x4v3 model.
                                    0 for weak denoise (keep noise), 1 for strong denoise ability.
    :param tile_w:                  Tile width. As too large images result in the out of GPU memory issue, so this tile
                                    option will first crop input images into tiles, and then process each of them.
                                    Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_h:                  Tile height.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("realesrgan: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("realesrgan: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("realesrgan: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("realesrgan: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("realesrgan: setting num_streams greater than `core.num_threads` is useless")

    if trt:
        if nvfuser:
            raise vs.Error("realesrgan: nvfuser and trt are mutually exclusive")

        if cuda_graphs:
            raise vs.Error("realesrgan: cuda_graphs and trt are mutually exclusive")

    if model not in range(6):
        raise vs.Error("realesrgan: model must be 0, 1, 2, 3, 4, or 5")

    if denoise_strength < 0 or denoise_strength > 1:
        raise vs.Error("realesrgan: denoise_strength must be between 0.0 and 1.0 (inclusive)")

    if os.path.getsize(os.path.join(model_dir, "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth")) == 0:
        raise vs.Error("realesrgan: model files have not been downloaded. run 'python -m vsrealesrgan' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    if model_path is None:
        match model:
            case 0:  # x4 RRDBNet model
                model_name = "ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"
                module = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                scale = 4
            case 1:  # x2 RRDBNet model
                model_name = "RealESRGAN_x2plus.pth"
                module = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                scale = 2
            case 2:  # x4 RRDBNet model
                model_name = "RealESRGAN_x4plus.pth"
                module = RRDBNet(3, 3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                scale = 4
            case 3:  # x4 RRDBNet model with 6 blocks
                model_name = "RealESRGAN_x4plus_anime_6B.pth"
                module = RRDBNet(3, 3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
                scale = 4
            case 4:  # x4 VGG-style model (XS size)
                model_name = "realesr-animevideov3.pth"
                module = SRVGGNetCompact(3, 3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
                scale = 4
            case 5:  # x4 VGG-style model (S size)
                model_name = "realesr-general-x4v3.pth"
                module = SRVGGNetCompact(3, 3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")
                scale = 4

        model_path = os.path.join(model_dir, model_name)

        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            loadnet = loadnet["params_ema"]
        elif "params" in loadnet:
            loadnet = loadnet["params"]
    else:
        model_path = os.path.realpath(model_path)
        model_name = os.path.basename(model_path)

        loadnet = torch.load(model_path, map_location="cpu")
        if "params_ema" in loadnet:
            loadnet = loadnet["params_ema"]
        elif "params" in loadnet:
            loadnet = loadnet["params"]

        if "conv_first.weight" in loadnet:
            num_feat = loadnet["conv_first.weight"].shape[0]
            num_block = int(list(loadnet)[-11].split(".")[1]) + 1
            num_grow_ch = loadnet["body.0.rdb1.conv1.weight"].shape[0]

            match loadnet["conv_first.weight"].shape[1]:
                case 48:
                    scale = 1
                case 12:
                    scale = 2
                case _:
                    scale = 4

            module = RRDBNet(3, 3, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, scale=scale)
        else:
            num_feat = loadnet["body.0.weight"].shape[0]
            num_conv = int(list(loadnet)[-1].split(".")[1]) // 2 - 1
            scale = math.isqrt(loadnet[list(loadnet)[-1]].shape[0] // 3)
            module = SRVGGNetCompact(3, 3, num_feat=num_feat, num_conv=num_conv, upscale=scale, act_type="prelu")

    if model == 5 and denoise_strength != 1:
        wdn_model_path = model_path.replace("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        dni_weight = [denoise_strength, 1 - denoise_strength]

        net_b = torch.load(wdn_model_path, map_location="cpu")
        if "params_ema" in net_b:
            net_b = net_b["params_ema"]
        elif "params" in net_b:
            net_b = net_b["params"]

        for k, v in loadnet.items():
            loadnet[k] = dni_weight[0] * v + dni_weight[1] * net_b[k]

    module.load_state_dict(loadnet)
    module.eval().to(device, memory_format=torch.channels_last)
    if fp16:
        module.half()

    match scale:
        case 1:
            modulo = 4
        case 2:
            modulo = 2
        case _:
            modulo = 1

    if tile_w > 0 and tile_h > 0:
        pad_w = math.ceil(min(tile_w + 2 * tile_pad, clip.width) / modulo) * modulo
        pad_h = math.ceil(min(tile_h + 2 * tile_pad, clip.height) / modulo) * modulo
    else:
        pad_w = math.ceil(clip.width / modulo) * modulo
        pad_h = math.ceil(clip.height / modulo) * modulo

    if nvfuser:
        module = memory_efficient_fusion(module)

    if cuda_graphs:
        graph: list[torch.cuda.CUDAGraph] = []
        static_input: list[torch.Tensor] = []
        static_output: list[torch.Tensor] = []

        for i in range(num_streams):
            static_input.append(
                torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device).to(memory_format=torch.channels_last)
            )

            torch.cuda.synchronize(device=device)
            stream[i].wait_stream(torch.cuda.current_stream(device=device))
            with torch.cuda.stream(stream[i]):
                module(static_input[i])
            torch.cuda.current_stream(device=device).wait_stream(stream[i])
            torch.cuda.synchronize(device=device)

            graph.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graph[i], stream=stream[i]):
                static_output.append(module(static_input[i]))

        backend = Backend.CUDAGraphs(graph, static_input, static_output)
    elif trt:
        device_name = torch.cuda.get_device_name(device)
        trt_version = tensorrt.__version__
        dimensions = f"{pad_w}x{pad_h}"
        precision = "fp16" if fp16 else "fp32"
        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_path),
            (
                f"{model_name}"
                + f"_{device_name}"
                + f"_trt-{trt_version}"
                + f"_{dimensions}"
                + f"_{precision}"
                + f"_workspace-{trt_max_workspace_size}"
                + (f"_denoise-{denoise_strength}" if model == 5 else "")
                + ".pt"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            lower_setting = LowerSetting(
                lower_precision=LowerPrecision.FP16 if fp16 else LowerPrecision.FP32,
                min_acc_module_size=1,
                max_workspace_size=trt_max_workspace_size,
                dynamic_batch=False,
                tactic_sources=1 << int(tensorrt.TacticSource.EDGE_MASK_CONVOLUTIONS)
                | 1 << int(tensorrt.TacticSource.JIT_CONVOLUTIONS),
            )
            lowerer = Lowerer.create(lower_setting=lower_setting)
            module = lowerer(
                module,
                [torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device).to(memory_format=torch.channels_last)],
            )
            torch.save(module, trt_engine_path)

        del module
        torch.cuda.empty_cache()
        module = [torch.load(trt_engine_path) for _ in range(num_streams)]
        backend = Backend.TensorRT(module)
    else:
        backend = Backend.Eager(module)

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_tensor(f[0], device)

            if tile_w > 0 and tile_h > 0:
                output = tile_process(img, scale, tile_w, tile_h, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "reflect")

                if cuda_graphs:
                    static_input[local_index].copy_(img)
                    graph[local_index].replay()
                    output = static_output[local_index]
                elif trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                output = output[:, :, : h * scale, : w * scale]

            return tensor_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return torch.from_numpy(array).unsqueeze(0).to(device, memory_format=torch.channels_last).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = tensor.squeeze(0).detach().cpu().numpy()
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile_w: int,
    tile_h: int,
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.CUDAGraphs | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_w)
    tiles_y = math.ceil(height / tile_h)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_w
            ofs_y = y * tile_h

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_w, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_h, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            h, w = input_tile.shape[2:]
            mode = "reflect" if pad_w - w < w and pad_h - h < h else "replicate"
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), mode)

            # process tile
            if isinstance(backend, Backend.CUDAGraphs):
                backend.static_input[index].copy_(input_tile)
                backend.graph[index].replay()
                output_tile = backend.static_output[index]
            elif isinstance(backend, Backend.TensorRT):
                output_tile = backend.module[index](input_tile)
            else:
                output_tile = backend.module(input_tile)

            output_tile = output_tile[:, :, : h * scale, : w * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile[
                :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile
            ]

    return output
