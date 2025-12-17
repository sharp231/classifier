from PIL import Image, ImageOps
import numpy as np

def preprocess(filepath, img_size=32, crop_right_ratio=1.0):
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)   # EXIF回転反映
    img = img.convert("RGB")

    # 文字バナー対策（必要なら右側を切る）
    if crop_right_ratio < 1.0:
        w, h = img.size
        img = img.crop((0, 0, int(w * crop_right_ratio), h))

    # 縦横比を保って中央を切り抜き
    img = ImageOps.fit(img, (img_size, img_size), method=Image.Resampling.LANCZOS)

    x = np.asarray(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, 0)
    return x