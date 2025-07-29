# === app.py  ― 2025‑07‑29 fix: 正しい v3 import ==========================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI

# ★ Messaging 本体
from linebot.v3.messaging import (
    Configuration, MessagingApi,
    ReplyMessageRequest,
)

# ★ UI コンポーネント類は “.models” 配下！
from linebot.v3.messaging.models import (
    TextMessage, ImageMessage,
    QuickReply, QuickReplyItem, MessageAction,
    FlexMessage, FlexBubble, FlexCarousel,
    BoxComponent, TextComponent, ImageComponent,
)

# ★ Webhook
from linebot.v3.webhooks import (
    WebhookHandler, MessageEvent,
    TextMessageContent, ImageMessageContent,
)

import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
# ---------------------------------------------------------------------

load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

# ---- init -----------------------------------------------------------
api_cfg = Configuration(access_token=CHAN_TOKEN)
api     = MessagingApi(api_cfg)
oai     = OpenAI(api_key=OPENAI_KEY)
app     = Flask(__name__)
handler = WebhookHandler(CHAN_SECRET)

# ---- k‑NN -----------------------------------------------------------
df  = pd.read_csv("recipes.csv")                   # Name, L, a, b, formula …
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)

state = defaultdict(dict)                          # user_id → {step,img,lv}

def extract_lab(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく一言で解説して。\n処方: {formula}\n40文字以内、日本語。"
    try:
        rsp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.7
        )
        return rsp.choices[0].message.content.strip()
    except Exception:
        traceback.print_exc()
        return "(解説取得エラー)"

def make_bubble(rec) -> FlexBubble:
    return FlexBubble(
        hero=ImageComponent(
            url=f"{CHIP_BASE}/{rec.Name}.png",
            size="full", aspect_ratio="1:1", aspect_mode="cover"
        ),
        body=BoxComponent(
            layout="vertical", spacing="sm",
            contents=[
                TextComponent(text=rec.Name, weight="bold", size="md"),
                TextComponent(text=rec.formula, wrap=True, size="sm"),
                TextComponent(text=gpt_comment(rec.formula), wrap=True,
                              size="sm", color="#888888")
            ]
        )
    )

# ---------------------------------------------------------------------
@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, sig)
    except Exception:
        abort(400)
    return "OK", 200

# ---------- 画像 -----------------------------------------------------
@handler.add(MessageEvent, message=ImageMessageContent)
def on_image(ev):
    img_bytes = api.get_message_content(ev.message.id).body
    uid = ev.source.user_id
    state[uid] = {"step": "ask_lv", "img": img_bytes}
    api.reply_message(
        ReplyMessageRequest(
            reply_token=ev.reply_token,
            messages=[TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")]
        )
    )

# ---------- テキスト --------------------------------------------------
@handler.add(MessageEvent, message=TextMessageContent)
def on_text(ev):
    txt = ev.message.text.strip()
    uid = ev.source.user_id
    st  = state.get(uid, {})

    # ① LV
    if st.get("step") == "ask_lv":
        try:
            lv = int(txt); assert 0 <= lv <= 19
        except Exception:
            api.reply_message(
                ReplyMessageRequest(ev.reply_token,
                                    [TextMessage(text="0〜19 の数字で送ってね❗")])
            )
            return
        st["lv"] = lv; st["step"] = "ask_hist"
        qr = QuickReply(items=[
            QuickReplyItem(action=MessageAction(label=l, text=f"HIST:{v}"))
            for l, v in [("0回","0"),("1回","1"),("2回","2"),
                         ("3回以上","3"),("縮毛","S"),("パーマ","P")]
        ])
        api.reply_message(
            ReplyMessageRequest(ev.reply_token,
                                [TextMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=qr)])
        )
        return

    # ② 履歴→レシピ
    if txt.startswith("HIST:") and st.get("step") == "ask_hist":
        hist = txt.split(":")[1]; lv = st["lv"]; _ = extract_lab(st["img"])
        df["score"] = (df["L"]-lv*12).abs()*0.5 + (df["formula"].str.contains("6%")&(hist=="S"))*10
        top3 = df.nsmallest(3,"score")
        car = FlexCarousel(contents=[make_bubble(r) for r in top3.itertuples()])
        api.reply_message(
            ReplyMessageRequest(ev.reply_token,
                                [FlexMessage(alt_text="おすすめレシピ", contents=car)])
        )
        state.pop(uid, None)

# ---------- healthcheck ----------------------------------------------
@app.route("/callback", methods=["GET"])
def health(): return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
# =====================================================================
