from PIL import Image, ImageDraw, ImageFilter
import numpy as np, os, cv2, colorsys

def desaturate(rgb, sat_ratio=0.75):
    """RGB(0-255) → HLS → 彩度を sat_ratio 倍にして戻す"""
    r,g,b = [x/255 for x in rgb]
    h,l,s = colorsys.rgb_to_hls(r,g,b)
    r2,g2,b2 = colorsys.hls_to_rgb(h, l, s*sat_ratio)
    return tuple(int(x*255) for x in (r2,g2,b2))

def make_chip(rgb, out_path, size=256, radius=40):
    # ① ベース色を少しくすませて作成
    base = Image.new("RGB", (size, size), desaturate(rgb, 0.75))

    # ② 髪っぽい縦筋ノイズを作る（強めVer.）
    noise = np.random.normal(0, 40, (size, size)).astype('float32')

    STRIPE = 50  # ← 太さ：10=細 30=太　好みで調整
    kernel = np.ones((STRIPE, 1), np.float32) / STRIPE
    noise_v = cv2.filter2D(noise, -1, kernel)             # 縦方向にブラー
    noise_rgb = np.clip(noise_v[..., None] + 128, 0, 255) \
                    .astype('uint8').repeat(3, 2)

    noise_img = Image.fromarray(noise_rgb)
    hair = Image.blend(base, noise_img, alpha=0.55)        # ← α を強めに

    # ③ 角丸マスクで仕上げ
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size, size), radius=radius, fill=255)
    chip = Image.composite(hair, Image.new("RGB", (size, size)), mask)
    chip.save(out_path)
# --- カラーパレット（くすみ気味に調整してOK） -------------------
palette = {
    "Beige-10LV": (205, 185, 150),
    "Gray-8LV"  : (120, 120, 120),
    "Pink-8LV"  : (220, 150, 175),
    "Green-9LV" : ( 85, 160, 110),
    "Blue-9LV"  : ( 70, 110, 200),
}

os.makedirs("static_img", exist_ok=True)
for name, rgb in palette.items():
    make_chip(rgb, f"static_img/{name}.png")

print("✅ 髪質チップを生成しました！")
