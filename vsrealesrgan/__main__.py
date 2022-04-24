import os.path as osp

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(osp.join(osp.dirname(__file__), filename), 'wb') as f:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename, total=int(r.headers.get('content-length', 0))) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == '__main__':
    download_model('https://github.com/HolyWu/vs-realesrgan/releases/download/model/realesr-animevideov3.onnx')
    download_model('https://github.com/HolyWu/vs-realesrgan/releases/download/model/RealESRGAN_x2plus.onnx')
    download_model('https://github.com/HolyWu/vs-realesrgan/releases/download/model/RealESRGAN_x4plus.onnx')
    download_model('https://github.com/HolyWu/vs-realesrgan/releases/download/model/RealESRGAN_x4plus_anime_6B.onnx')
