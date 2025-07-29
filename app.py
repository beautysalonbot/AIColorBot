# === app.py ── LINE SDK v3 + Flex (修正版) ===============================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# ---------- LINE SDK v3 --------------------------------------------------
from linebot.v3.messaging import Configuration, MessagingApi
from linebot.v3.messaging.models import (
    ReplyMessageRequest,
    TextMessage, QuickReply, QuickReplyItem, MessageAction,
    FlexMessage, FlexBubble, FlexCarousel,
    BoxComponent, TextComponent, ImageComponent,
)
from linebot.v3.webhooks import WebhookHandler
from linebot.v3.webhooks.models import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent,
)
# ------------------------------------------------------------------------

# ---------- config & init -----------------------------------------------
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

openai_client = OpenAI(api_key=OPENAI_KEY)

cfg   = Configuration(access_token=CHAN_TOKEN)
api   = MessagingApi(cfg)
app   = Flask(__name__)
parser = WebhookHandler(CHAN_SECRET)

# ---------- k‑NN ---------------------------------------------------------
df  = pd.read_csv("recipes.csv")                 # Name, L, a, b, formula …
knn = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)

state = defaultdict(dict)                        # user_id → {step,img,lv}

# ---------- helpers ------------------------------------------------------
def extract_lab(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく一言で解説して。\n処方: {formula}\n40文字以内、日本語。"
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=60, temperature=0.7
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("GPT Error:", type(e).__name__, "-", e, file=sys.stderr)
        traceback.print_exc()
        return "(解説取得エラー)"

def make_bubble(rec) -> FlexBubble:
    return FlexBubble(
        hero=ImageComponent(
            url=f"{CHIP_BASE}/{rec.Name}.png",
            size="full", aspect_mode="cover", aspect_ratio="1:1"
        ),
        body=BoxComponent(
            layout="vertical", spacing="sm",
            contents=[
                TextComponent(text=rec.Name, weight="bold", size="md"),
                TextComponent(text=rec.formula, size="sm", wrap=True),
                TextComponent(text=gpt_comment(rec.formula),
                              size="sm", color="#888888", wrap=True)
            ]
        )
    )

def reply_text(token: str, text: str):
    api.reply_message(ReplyMessageRequest(
        reply_token=token,
        messages=[TextMessage(text=text)]
    ))

# =========================== Webhook =====================================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body      = request.get_data(as_text=True)
    try:
        events = parser.parse(body, signature)
    except Exception:
        abort(400)

    for ev in events:

        # ---------- ① 画像 ----------
        if isinstance(ev, MessageEvent) and isinstance(ev.message, ImageMessageContent):
            img_obj   = api.get_message_content(ev.message.id)
            img_bytes = img_obj.body if hasattr(img_obj, "body") else img_obj
            uid = ev.source.user_id
            state[uid] = {"step":"ask_lv", "img":img_bytes}
            reply_text(ev.reply_token, "現在の明度を 0〜19 の数字で送ってください📩")
            return "OK", 200

        # ---------- ② テキスト ----------
        if isinstance(ev.message, TextMessageContent):
            txt  = ev.message.text.strip()
            uid  = ev.source.user_id
            info = state.get(uid, {})

            # --- LV 受付 ---
            if info.get("step") == "ask_lv":
                try:
                    lv = int(txt);  assert 0<=lv<=19
                except Exception:
                    reply_text(ev.reply_token, "0〜19 の数字で送ってね❗")
                    return "OK", 200

                info["lv"]   = lv
                info["step"] = "ask_hist"

                qr = QuickReply(items=[
                    QuickReplyItem(action=MessageAction(label="0回",    text="HIST:0")),
                    QuickReplyItem(action=MessageAction(label="1回",    text="HIST:1")),
                    QuickReplyItem(action=MessageAction(label="2回",    text="HIST:2")),
                    QuickReplyItem(action=MessageAction(label="3回以上", text="HIST:3")),
                    QuickReplyItem(action=MessageAction(label="縮毛",    text="HIST:S")),
                    QuickReplyItem(action=MessageAction(label="パーマ",  text="HIST:P")),
                ])
                api.reply_message(ReplyMessageRequest(
                    reply_token=ev.reply_token,
                    messages=[TextMessage(
                        text="ブリーチ・縮毛などの履歴を選んでね",
                        quick_reply=qr
                    )]
                ))
                return "OK", 200

            # --- 履歴を受信したら推論＋GPT -------------
            if txt.startswith("HIST:") and info.get("step") == "ask_hist":
                hist = txt.split(":",1)[1]
                lv   = info["lv"]

                df["score"] = (df["L"]-lv*12).abs()*0.5 + \
                              (df["formula"].str.contains("6%")&(hist=="S"))*10
                top3 = df.nsmallest(3, "score")

                carousel = FlexCarousel(contents=[make_bubble(r) for r in top3.itertuples()])
                api.reply_message(ReplyMessageRequest(
                    reply_token=ev.reply_token,
                    messages=[FlexMessage(alt_text="おすすめレシピ", contents=carousel)]
                ))
                state.pop(uid, None)
                return "OK", 200

    return "OK", 200  # fallback

# ---------- Render health-check ------------------------------------------
@app.route("/callback", methods=["GET"])
def health():
    return "OK", 200

# ---------- local run ----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
# ========================================================================
