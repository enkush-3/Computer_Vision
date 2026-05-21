import os
from PIL import Image


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}


def preprocess_images(input_dir: str, output_dir: str, resize_factor: int = 1) -> int:
    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Зургийн хавтас олдсонгүй: {input_dir}\n"
            f"   image_dir0 замыг зөв оруулна уу."
        )

    image_files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]
    n_images = len(image_files)

    if n_images < 2:
        raise ValueError(
            f"Зураг хангалтгүй: {n_images} зураг олдлоо. Дор хаяж 2 шаардлагатай."
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Оролтын зураг   : {input_dir}")
    print(f"Боловсруулсан   : {output_dir}/")
    print(f"Resize factor   : 1/{resize_factor} {'(original)' if resize_factor == 1 else ''}")
    print(f"Нийт {n_images} зураг олдлоо")
    print(f"Боловсруулж эхэллээ (resize 1/{resize_factor})...")

    for idx, filename in enumerate(image_files, 1):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        img = Image.open(src_path)
        w, h = img.size

        if resize_factor > 1:
            img = img.resize((w // resize_factor, h // resize_factor))

        img.save(dst_path)

        if idx % 10 == 0 or idx == n_images:
            print(f"   [{idx}/{n_images}] {filename} ({w}x{h} → {w//resize_factor}x{h//resize_factor})")

    print(f"\nНийт {n_images} зураг боловсруулагдлаа → '{output_dir}/' хавтаст хадгалагдлаа")
    return n_images
