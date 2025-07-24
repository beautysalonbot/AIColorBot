# === app.py (Flex カルーセル対応・完成版) =============================
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookParser
from linebot.models import (
    MessageEvent, ImageMessage, TextMessage,
    TextSendMessage, FlexSendMessage
)
from dotenv import load_dotenv
from openai import OpenAI
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# ---------- 環境変数 ----------
load_dotenv()
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")
# 静的チップ画像を置いた Render Static の URL
CHIP_BASE      = os.getenv("CHIP_BASE",
                           "https://aic-olorbot-static.onrender.com")

# ---------- 初期化 ----------
client   = OpenAI(api_key=OPENAI_KEY)
app      = Flask(__name__)
bot      = LineBotApi(CHANNEL_TOKEN)
parser   = WebhookParser(CHANNEL_SECRET)

# ---------- k‑NN 用データ ----------
df  = pd.read_csv("recipes.csv")            # name,L,a,b,formula 列を想定
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)

user_state = defaultdict(dict)              # uid → {step, img, lv}

# ---------- 画像 bytes → 平均 LAB ----------
def extract_lab(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)

# ---------- GPT で 40 文字解説 ----------
def gpt_comment(formula: str) -> str:
    prompt = (f"以下のヘアカラー処方を美容師らしく一言で解説して。\n"
              f"処方: {formula}\n40文字以内、日本語。")
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.7
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("GPT Error:", type(e).__name__, "-", e, file=sys.stderr)
        traceback.print_exc()
        return "(解説取得エラー)"

# ====================================================================
@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        events = parser.parse(body, sig)
    except Exception:
        abort(400)

    for ev in events:
        # ---------- ① 画像 ----------
        if isinstance(ev, MessageEvent) and isinstance(ev.message, ImageMessage):
            content = bot.get_message_content(ev.message.id)
            uid = ev.source.user_id
            user_state[uid] = {"step": "ask_lv", "img": content.content}
            bot.reply_message(ev.reply_token,
                TextSendMessage(text="現在の明度を 0〜19 の数字で送ってください📩"))
            continue

        # ---------- ② テキスト ----------
        if isinstance(ev, MessageEvent) and isinstance(ev.message, TextMessage):
            text = ev.message.text
            uid  = ev.source.user_id
            st   = user_state.get(uid, {})

            # --- LV を受信 ---
            if st.get("step") == "ask_lv":
                try:
                    lv = int(text)
                    assert 0 <= lv <= 19
                except Exception:
                    bot.reply_message(ev.reply_token,
                        TextSendMessage(text="0〜19 の数字で送ってね❗"))
                    continue
                st["lv"] = lv
                st["step"] = "ask_hist"
                bot.reply_message(ev.reply_token,
                    TextSendMessage(text="ブリーチ履歴を 0/1/2/S で返信してね\n(0=なし,S=縮毛)"))
                continue

            # --- 履歴を受信 → 推論＋Flex 返信 ---
            if text.startswith("HIST:") and st.get("step") == "ask_hist":
                hist = text.split(":")[1]
                lv   = st["lv"]
                lab  = extract_lab(st["img"])

                # k‑NN スコアリング
                df["lv_diff"]  = (df["L"] - lv * 12).abs()
                df["hist_pen"] = (df["formula"].str.contains("6%") & (hist == "S")).astype(int)
                df["score"]    = df["lv_diff"] * 0.5 + df["hist_pen"] * 10
                top3 = df.nsmallest(3, "score")

                # Flex Bubble を組み立て
                bubbles = []
                for _, row in top3.iterrows():
                    bubbles.append({
                        "type": "bubble",
                        "hero": {
                            "type": "image",
                            "url": f"{CHIP_BASE}/{row['name']}.png",
                            "size": "full",
                            "aspectRatio": "1:1",
                            "aspectMode": "cover"
                        },
                        "body": {
                            "type": "box",
                            "layout": "vertical",
                            "spacing": "sm",
                            "contents": [
                                {"type": "text", "text": row["name"],
                                 "weight": "bold", "size": "md"},
                                {"type": "text", "text": row["formula"],
                                 "wrap": True, "size": "sm"},
                                {"type": "text", "text": gpt_comment(row["formula"]),
                                 "wrap": True, "size": "sm", "color": "#888888"}
                            ]
                        }
                    })

                flex = FlexSendMessage(
                    alt_text="おすすめレシピ",
                    contents={"type": "carousel", "contents": bubbles}
                )
                bot.reply_message(ev.reply_token, flex)
                user_state.pop(uid, None)
                continue

    return "OK", 200

# ---------- LINE Verify 用 ----------
@app.route("/callback", methods=["GET"])
def health():
    return "OK", 200

# ---------- ローカル起動 ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# ====================================================================
