from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from enum import IntEnum
from threading import Lock

import numpy as np
import torch
import torch.nn.functional as F
import vapoursynth as vs

from .rrdbnet_arch import RRDBNet
from .srvgg_arch import SRVGGNetCompact

__version__ = "5.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

warnings.filterwarnings("ignore", "The given NumPy array is not writable")

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]


class RealESRGANModel(IntEnum):
    ESRGAN_SRx4 = 0
    RealESRGAN_x2plus = 1
    RealESRGAN_x4plus = 2
    RealESRGAN_x4plus_anime_6B = 3
    realesr_animevideov3 = 4
    realesr_general_x4v3 = 5

    AnimeJaNai_HD_V3_Compact_2x = 100
    AnimeJaNai_HD_V3_UltraCompact_2x = 101
    AnimeJaNai_HD_V3_SuperUltraCompact_2x = 102

    AniScale2_Compact_2x = 200
    AniScale2_Refiner_1x = 201
    OpenProteus_Compact_2x = 202
    Ani4Kv2_Compact_2x = 203
    Ani4Kv2_UltraCompact_2x = 204


@torch.inference_mode()
def realesrgan(
    clip: vs.VideoNode,
    device_index: int = 0,
    num_streams: int = 1,
    model: RealESRGANModel = RealESRGANModel.AnimeJaNai_HD_V3_UltraCompact_2x,
    model_path: str | None = None,
    denoise_strength: float = 0.5,
    tile: list[int] = [0, 0],
    tile_pad: int = 8,
    trt: bool = False,
    trt_debug: bool = False,
    trt_workspace_size: int = 0,
    trt_int8: bool = False,
    trt_int8_sample_step: int = 72,
    trt_int8_batch_size: int = 1,
    trt_cache_dir: str = model_dir,
) -> vs.VideoNode:
    """Training Real-World Blind Super-Resolution with Pure Synthetic Data

    :param clip:                    Clip to process. Only RGBH and RGBS formats are supported. RGBH performs inference
                                    in FP16 mode while RGBS performs inference in FP32 mode, except `trt_int8=True`.
    :param device_index:            Device ordinal of the GPU.
    :param num_streams:             Number of CUDA streams to enqueue the kernels.
    :param model:                   Model to use.
    :param model_path:              Path to custom model file.
    :param denoise_strength:        Denoise strength for realesr-general-x4v3 model.
                                    0 for weak denoise (keep noise), 1 for strong denoise ability.
    :param tile:                    Tile width and height. As too large images result in the out of GPU memory issue, so
                                    this tile option will first crop input images into tiles, and then process each of
                                    them. Finally, they will be merged into one image. 0 denotes for do not use tile.
    :param tile_pad:                Pad size for each tile, to remove border artifacts.
    :param trt:                     Use TensorRT for high-performance inference.
    :param trt_debug:               Print out verbose debugging information.
    :param trt_workspace_size:      Size constraints of workspace memory pool.
    :param trt_int8:                Perform inference in INT8 mode using Post Training Quantization (PTQ). Calibration
                                    datasets are sampled from input clip while building the engine.
    :param trt_int8_sample_step:    Interval between sampled frames.
    :param trt_int8_batch_size:     How many samples per batch to load. Calibrate with as large a single batch as
                                    possible. Batch size can affect truncation error and may impact the final result.
    :param trt_cache_dir:           Directory for TensorRT engine file. Engine will be cached when it's built for the
                                    first time. Note each engine is created for specific settings such as model
                                    path/name, precision, workspace etc, and specific GPUs and it's not portable.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("realesrgan: this is not a clip")

    if clip.format.id not in [vs.RGBH, vs.RGBS]:
        raise vs.Error("realesrgan: only RGBH and RGBS formats are supported")

    if not torch.cuda.is_available():
        raise vs.Error("realesrgan: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("realesrgan: num_streams must be at least 1")

    if model not in RealESRGANModel:
        raise vs.Error("realesrgan: model must be one of the members in RealESRGANModel")

    if denoise_strength < 0 or denoise_strength > 1:
        raise vs.Error("realesrgan: denoise_strength must be between 0.0 and 1.0 (inclusive)")

    if not isinstance(tile, list) or len(tile) != 2:
        raise vs.Error("realesrgan: tile must be a list with 2 items")

    if trt and trt_int8 and clip.format.bits_per_sample != 32:
        raise vs.Error("realesrgan: INT8 mode only supports RGBS format")

    if trt_int8_sample_step < 1:
        raise vs.Error("realesrgan: trt_int8_sample_step must be at least 1")

    if os.path.getsize(os.path.join(model_dir, "Ani4Kv2_Compact_2x.pth")) == 0:
        raise vs.Error("realesrgan: model files have not been downloaded. run 'python -m vsrealesrgan' first")

    torch.set_float32_matmul_precision("high")

    fp16 = clip.format.bits_per_sample == 16
    dtype = torch.half if fp16 else torch.float

    device = torch.device("cuda", device_index)

    inf_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    f2t_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]
    t2f_streams = [torch.cuda.Stream(device) for _ in range(num_streams)]

    inf_stream_locks = [Lock() for _ in range(num_streams)]
    f2t_stream_locks = [Lock() for _ in range(num_streams)]
    t2f_stream_locks = [Lock() for _ in range(num_streams)]

    if model_path is None:
        model_name = f"{RealESRGANModel(model).name}.pth"
        model_path = os.path.join(model_dir, model_name)
    else:
        model_path = os.path.realpath(model_path)
        model_name = os.path.basename(model_path)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if "params_ema" in state_dict:
        state_dict = state_dict["params_ema"]
    elif "params" in state_dict:
        state_dict = state_dict["params"]

    if model == RealESRGANModel.realesr_general_x4v3 and denoise_strength != 1:
        wdn_model_path = model_path.replace("realesr_general_x4v3", "realesr_general_wdn_x4v3")
        dni_weight = [denoise_strength, 1 - denoise_strength]

        net_b = torch.load(wdn_model_path, map_location=device, weights_only=True)
        if "params_ema" in net_b:
            net_b = net_b["params_ema"]
        elif "params" in net_b:
            net_b = net_b["params"]

        for k, v in state_dict.items():
            state_dict[k] = dni_weight[0] * v + dni_weight[1] * net_b[k]

    if "conv_first.weight" in state_dict:
        num_feat = state_dict["conv_first.weight"].shape[0]
        num_block = int(list(state_dict)[-11].split(".")[1]) + 1
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]

        match state_dict["conv_first.weight"].shape[1]:
            case 48:
                scale = 1
            case 12:
                scale = 2
            case _:
                scale = 4

        with torch.device("meta"):
            module = RRDBNet(3, 3, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch, scale=scale)
    else:
        num_feat = state_dict["body.0.weight"].shape[0]
        num_conv = int(list(state_dict)[-1].split(".")[1]) // 2 - 1
        scale = math.isqrt(state_dict[list(state_dict)[-1]].shape[0] // 3)

        with torch.device("meta"):
            module = SRVGGNetCompact(3, 3, num_feat=num_feat, num_conv=num_conv, upscale=scale, act_type="prelu")

    module.load_state_dict(state_dict, assign=True)
    module.eval().to(device)
    if fp16:
        module.half()

    match scale:
        case 1:
            modulo = 4
        case 2:
            modulo = 2
        case _:
            modulo = 1

    if all(t > 0 for t in tile):
        pad_w = math.ceil(min(tile[0] + 2 * tile_pad, clip.width) / modulo) * modulo
        pad_h = math.ceil(min(tile[1] + 2 * tile_pad, clip.height) / modulo) * modulo
    else:
        pad_w = math.ceil(clip.width / modulo) * modulo
        pad_h = math.ceil(clip.height / modulo) * modulo

    if trt:
        import tensorrt
        import torch_tensorrt
        import torch_tensorrt.ts.logging as logging
        from torch.utils.data import DataLoader, Dataset
        from torch_tensorrt.ts.ptq import DataLoaderCalibrator

        class MyDataset(Dataset):
            def __init__(self, clip: vs.VideoNode, device: torch.device) -> None:
                super().__init__()
                self.clip = clip
                self.device = device

            def __getitem__(self, index: int) -> torch.Tensor:
                with self.clip.get_frame(index * trt_int8_sample_step) as f:
                    return frame_to_tensor(f, self.device)

            def __len__(self) -> int:
                return math.ceil(self.clip.num_frames / trt_int8_sample_step)

        logging.set_reportable_log_level(logging.Level.Debug if trt_debug else logging.Level.Info)
        logging.set_is_colored_output_on(True)

        trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{model_name}"
                + f"_{pad_w}x{pad_h}"
                + f"_{'int8' if trt_int8 else 'fp16' if fp16 else 'fp32'}"
                + (f"_denoise-{denoise_strength}" if model == RealESRGANModel.realesr_general_x4v3 else "")
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + ".ts"
            ),
        )

        if not os.path.isfile(trt_engine_path):
            inputs = [torch.zeros((1, 3, pad_h, pad_w), dtype=dtype, device=device)]
            module = torch.jit.trace(module, inputs)

            if trt_int8:
                dataset = MyDataset(clip, device)
                dataloader = DataLoader(dataset, batch_size=trt_int8_batch_size)
                calibrator = DataLoaderCalibrator(dataloader, device=device)

            module = torch_tensorrt.compile(
                module,
                ir="ts",
                inputs=inputs,
                enabled_precisions={torch.half, torch.int8} if trt_int8 else {dtype},
                device=torch_tensorrt.Device(gpu_id=device_index),
                workspace_size=trt_workspace_size,
                calibrator=calibrator if trt_int8 else None,
                truncate_long_and_double=True,
                min_block_size=1,
            )

            torch.jit.save(module, trt_engine_path)

        module = [torch.jit.load(trt_engine_path) for _ in range(num_streams)]
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

        with f2t_stream_locks[local_index], torch.cuda.stream(f2t_streams[local_index]):
            img = frame_to_tensor(f[0], device).unsqueeze(0)

            f2t_streams[local_index].synchronize()

        with inf_stream_locks[local_index], torch.cuda.stream(inf_streams[local_index]):
            if all(t > 0 for t in tile):
                output = tile_process(img, scale, tile, tile_pad, pad_w, pad_h, backend, local_index)
            else:
                h, w = img.shape[2:]
                img = F.pad(img, (0, pad_w - w, 0, pad_h - h), "replicate")

                if trt:
                    output = module[local_index](img)
                else:
                    output = module(img)

                output = output[:, :, : h * scale, : w * scale]

            inf_streams[local_index].synchronize()

        with t2f_stream_locks[local_index], torch.cuda.stream(t2f_streams[local_index]):
            return tensor_to_frame(output, f[1].copy(), t2f_streams[local_index])

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale, keep=True)
    return new_clip.std.FrameEval(
        lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
    )


def frame_to_tensor(frame: vs.VideoFrame, device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(frame[plane])).to(device, non_blocking=True)
            for plane in range(frame.format.num_planes)
        ]
    ).clamp(0.0, 1.0)


def tensor_to_frame(tensor: torch.Tensor, frame: vs.VideoFrame, stream: torch.cuda.Stream) -> vs.VideoFrame:
    tensor = tensor.squeeze(0).detach()
    tensors = [tensor[plane].to("cpu", non_blocking=True) for plane in range(frame.format.num_planes)]

    stream.synchronize()

    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), tensors[plane].numpy())
    return frame


def tile_process(
    img: torch.Tensor,
    scale: int,
    tile: list[int],
    tile_pad: int,
    pad_w: int,
    pad_h: int,
    backend: Backend.Eager | Backend.TensorRT,
    index: int,
) -> torch.Tensor:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = img.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile[0])
    tiles_y = math.ceil(height / tile[1])

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile[0]
            ofs_y = y * tile[1]

            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile[0], width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile[1], height)

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
            input_tile = F.pad(input_tile, (0, pad_w - w, 0, pad_h - h), "replicate")

            # process tile
            if isinstance(backend, Backend.TensorRT):
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
