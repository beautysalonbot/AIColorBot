import cv2, glob, os, numpy as np
candidates = glob.glob("hair.*")
if not candidates:
    print("hair.xxx が見つかりません！同じフォルダーに置いてね。")
    raise SystemExit
img_path = candidates[0]
print("読み込むファイル:", os.path.basename(img_path))
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
avg_rgb = img_rgb.reshape(-1, 3).mean(axis=0).astype(int)
lab_avg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
print("平均 RGB  :", avg_rgb)
print("平均 L*a*b:", lab_avg)
# 旧 : np.savetxt("latest_lab.txt", lab_avg, delimiter=",")
# ↓↓↓ 新しく書き換え ↓↓↓
with open("latest_lab.txt", "w") as f:
    f.write(",".join([str(x) for x in lab_avg.tolist()]))
print("LAB 値を latest_lab.txt に保存しました！")
