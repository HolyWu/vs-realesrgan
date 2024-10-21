import os

import requests
from tqdm import tqdm


def download_model(url: str) -> None:
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", filename), "wb") as f:
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename,
            total=int(r.headers.get("content-length", 0)),
        ) as pbar:
            for chunk in r.iter_content(chunk_size=4096):
                f.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    url = "https://github.com/HolyWu/vs-realesrgan/releases/download/model/"
    models = [
        "Ani4Kv2_Compact_2x",
        "Ani4Kv2_UltraCompact_2x",
        "AnimeJaNai_HD_V3_Compact_2x",
        "AnimeJaNai_HD_V3_SuperUltraCompact_2x",
        "AnimeJaNai_HD_V3_UltraCompact_2x",
        "AnimeJaNai_HD_V3Sharp1_Compact_2x",
        "AnimeJaNai_HD_V3Sharp1_SuperUltraCompact_2x",
        "AnimeJaNai_HD_V3Sharp1_UltraCompact_2x",
        "AnimeJaNai_SD_V1beta34_Compact_2x",
        "AnimeJaNai_V2_Compact_2x",
        "AnimeJaNai_V2_SuperUltraCompact_2x",
        "AnimeJaNai_V2_UltraCompact_2x",
        "AniScale2_Compact_2x",
        "AniScale2_Refiner_1x",
        "ESRGAN_SRx4",
        "OpenProteus_Compact_2x",
        "realesr_animevideov3",
        "realesr_general_wdn_x4v3",
        "realesr_general_x4v3",
        "RealESRGAN_x2plus",
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
    ]
    for model in models:
        download_model(url + model + ".pth")
