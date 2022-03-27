import math
import os.path as osp

import numpy as np
import onnxruntime as ort
import vapoursynth as vs

dir_name = osp.dirname(__file__)


def RealESRGAN(
    clip: vs.VideoNode,
    model: int = 0,
    tile_w: int = 0,
    tile_h: int = 0,
    tile_pad: int = 10,
    provider: int = 1,
    device_id: int = 0,
    trt_max_workspace_size: int = 1073741824,
    trt_fp16: bool = False,
    trt_engine_cache: bool = True,
    trt_engine_cache_path: str = dir_name,
    log_level: int = 2,
) -> vs.VideoNode:
    '''
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

    Parameters:
        clip: Clip to process. Only RGB format with float sample type of 32 bit depth is supported.

        model: Model to use.
            0 = RealESRGAN_x2plus (x2 model for general images)
            1 = RealESRGAN_x4plus (x4 model for general images)
            2 = RealESRGAN_x4plus_anime_6B (x4 model optimized for anime images)
            3 = RealESRGANv2-animevideo-xsx2 (x2 model optimized for anime videos)
            4 = RealESRGANv2-animevideo-xsx4 (x4 model optimized for anime videos)

        tile_w, tile_h: Tile width and height, respectively. As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image. 0 denotes for do not use tile.

        tile_pad: The pad size for each tile, to remove border artifacts.

        provider: The hardware platform to execute on.
            0 = Default CPU
            1 = NVIDIA CUDA (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)
            2 = NVIDIA TensorRT (https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements)
            3 = DirectML (https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#requirements)
            4 = AMD MIGraphX (https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

        device_id: The device ID.

        trt_max_workspace_size: Maximum workspace size for TensorRT engine.

        trt_fp16: Enable FP16 mode in TensorRT.

        trt_engine_cache: Enable TensorRT engine caching. The purpose of using engine caching is to save engine build time in the case that TensorRT may take
            long time to optimize and build engine. Engine will be cached when it's built for the first time so next time when new inference session is created
            the engine can be loaded directly from cache. In order to validate that the loaded engine is usable for current inference, engine profile is also
            cached and loaded along with engine. If current input shapes are in the range of the engine profile, the loaded engine can be safely used. Otherwise
            if input shapes are out of range, profile cache will be updated to cover the new shape and engine will be recreated based on the new profile (and
            also refreshed in the engine cache). Note each engine is created for specific settings such as model path/name, precision, workspace, profiles etc,
            and specific GPUs and it's not portable, so it's essential to make sure those settings are not changing, otherwise the engine needs to be rebuilt
            and cached again.

            Warning: Please clean up any old engine and profile cache files (.engine and .profile) if any of the following changes:
                Model changes (if there are any changes to the model topology, opset version, operators etc.)
                ORT version changes (i.e. moving from ORT version 1.8 to 1.9)
                TensorRT version changes (i.e. moving from TensorRT 7.0 to 8.0)
                Hardware changes (Engine and profile files are not portable and optimized for specific NVIDIA hardware)

        trt_engine_cache_path: Specify path for TensorRT engine and profile files if trt_engine_cache is true.

        log_level: Log severity level. Applies to session load, initialization, etc.
            0 = Verbose
            1 = Info
            2 = Warning
            3 = Error
            4 = Fatal
    '''
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RealESRGAN: this is not a clip')

    if clip.format.id != vs.RGBS:
        raise vs.Error('RealESRGAN: only RGBS format is supported')

    if model < 0 or model > 4:
        raise vs.Error('RealESRGAN: model must be 0, 1, 2, 3, or 4')

    if osp.getsize(osp.join(dir_name, 'RealESRGAN_x2plus.onnx')) == 0:
        raise vs.Error("RealESRGAN: model files have not been downloaded. run 'python -m vsrealesrgan' first")

    if model == 0:  # x2 RRDBNet model
        model_name = 'RealESRGAN_x2plus.onnx'
        scale = 2
    elif model == 1:  # x4 RRDBNet model
        model_name = 'RealESRGAN_x4plus.onnx'
        scale = 4
    elif model == 2:  # x4 RRDBNet model with 6 blocks
        model_name = 'RealESRGAN_x4plus_anime_6B.onnx'
        scale = 4
    elif model == 3:  # x2 VGG-style model (XS size)
        model_name = 'RealESRGANv2-animevideo-xsx2.onnx'
        scale = 2
    else:  # x4 VGG-style model (XS size)
        model_name = 'RealESRGANv2-animevideo-xsx4.onnx'
        scale = 4

    modulo = 2 if scale == 2 else 1

    model_path = osp.join(dir_name, model_name)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = log_level

    cuda_ep = ('CUDAExecutionProvider', dict(device_id=device_id))

    if provider <= 0:
        providers = ['CPUExecutionProvider']
    elif provider == 1:
        providers = [cuda_ep]
    elif provider == 2:
        providers = [
            (
                'TensorrtExecutionProvider',
                dict(
                    device_id=device_id,
                    trt_max_workspace_size=trt_max_workspace_size,
                    trt_fp16_enable=trt_fp16,
                    trt_engine_cache_enable=trt_engine_cache,
                    trt_engine_cache_path=trt_engine_cache_path,
                ),
            ),
            cuda_ep,
        ]
    elif provider == 3:
        sess_options.enable_mem_pattern = False
        providers = [('DmlExecutionProvider', dict(device_id=device_id))]
    else:
        providers = [('MIGraphXExecutionProvider', dict(device_id=device_id))]

    session = ort.InferenceSession(model_path, sess_options, providers)

    def realesrgan(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        img = frame_to_ndarray(f[0])

        if tile_w > 0 and tile_h > 0:
            output = tile_process(img, scale, tile_w, tile_h, tile_pad, modulo, session)
        elif img.shape[2] % modulo == 0 and img.shape[3] % modulo == 0:
            output = session.run(None, {'input': img})[0]
        else:
            output = mod_pad(img, modulo, session, scale)

        return ndarray_to_frame(output, f[1].copy())

    new_clip = clip.std.BlankClip(width=clip.width * scale, height=clip.height * scale)
    return new_clip.std.ModifyFrame(clips=[clip, new_clip], selector=realesrgan)


def frame_to_ndarray(frame: vs.VideoFrame) -> np.ndarray:
    array = np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes)])
    return np.expand_dims(array, axis=0)


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    array = np.squeeze(array, axis=0)
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[plane, :, :])
    return frame


def tile_process(img: np.ndarray, scale: int, tile_w: int, tile_h: int, tile_pad: int, modulo: int, session: ort.InferenceSession) -> np.ndarray:
    batch, channel, height, width = img.shape
    output_shape = (batch, channel, height * scale, width * scale)

    # start with black image
    output = np.zeros_like(img, shape=output_shape)

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

            # process tile
            if input_tile.shape[2] % modulo == 0 and input_tile.shape[3] % modulo == 0:
                output_tile = session.run(None, {'input': input_tile})[0]
            else:
                output_tile = mod_pad(input_tile, modulo, session, scale)

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


def mod_pad(img: np.ndarray, modulo: int, session: ort.InferenceSession, scale: int) -> np.ndarray:
    mod_pad_h, mod_pad_w = 0, 0
    h, w = img.shape[2:]

    if h % modulo != 0:
        mod_pad_h = modulo - h % modulo

    if w % modulo != 0:
        mod_pad_w = modulo - w % modulo

    img = np.pad(img, ((0, 0), (0, 0), (0, mod_pad_h), (0, mod_pad_w)), 'reflect')
    output = session.run(None, {'input': img})[0]
    return output[:, :, : h * scale, : w * scale]
