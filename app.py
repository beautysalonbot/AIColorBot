# === imports ===============================================================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI

from linebot.v3.messaging import (
    Configuration, MessagingApi,
    ReplyMessageRequest, TextMessage, FlexMessage,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.webhooks import WebhookHandler
from linebot.v3.webhooks.models import (
    MessageEvent, TextMessageContent, ImageMessageContent
)

# ─────────── other libs ───────────
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
# ===========================================================================
# 1. 環境変数
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE",
              "https://aic-olorbot-static.onrender.com")

# 2. SDK instance
client  = OpenAI(api_key=OPENAI_KEY)
api_cfg = Configuration(access_token=CHAN_TOKEN)
api     = MessagingApi(api_cfg)
app     = Flask(__name__)
handler = WebhookHandler(CHAN_SECRET)

# 3. k-NN 下準備
df   = pd.read_csv("recipes.csv")           # Name,L,a,b,formula,...
knn  = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)
state = defaultdict(dict)                   # user_id -> {step, lv, img}

# ─────────── util ───────────
def extract_lab(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく一言で解説して。\n処方: {formula}\n40文字以内、日本語。"
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.7
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("[GPT Error]", e, file=sys.stderr)
        traceback.print_exc()
        return "(解説取得エラー)"

def bubble_dict(rec) -> dict:
    """Flex Bubble を dict で返す（import 不要）"""
    return {
        "type": "bubble",
        "hero": {
            "type": "image",
            "url": f"{CHIP_BASE}/{rec.Name}.png",
            "size": "full",
            "aspectMode": "cover",
            "aspectRatio": "1:1"
        },
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "text", "text": rec.Name,     "weight": "bold", "size": "md"},
                {"type": "text", "text": rec.formula,  "wrap": True,     "size": "sm"},
                {"type": "text", "text": gpt_comment(rec.formula),
                 "wrap": True, "size": "sm", "color": "#888"}
            ]
        }
    }

# ─────────── webhook (Render も同じパス) ───────────
@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    sig  = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body, sig)
    except Exception as e:
        print("[Webhook Error]", e, file=sys.stderr); abort(400)
    return "OK", 200

# メッセージハンドラ
@handler.add(MessageEvent)
def on_event(ev: MessageEvent):
    uid = ev.source.user_id

    # ① 画像が来た
    if isinstance(ev.message, ImageMessageContent):
        raw = api.get_message_content(ev.message.id)
        state[uid] = {"step": "ask_lv", "img": raw}
        api.reply_message(ReplyMessageRequest(
            reply_token=ev.reply_token,
            messages=[TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")]
        ))
        return

    # ② テキストが来た
    if not isinstance(ev.message, TextMessageContent):
        return
    txt = ev.message.text.strip()
    st  = state.get(uid, {})

    # ---- 明度入力フェーズ ----
    if st.get("step") == "ask_lv":
        try:
            lv = int(txt); assert 0 <= lv <= 19
        except Exception:
            api.reply_message(ReplyMessageRequest(
                reply_token=ev.reply_token,
                messages=[TextMessage(text="0〜19 の数字で送ってね❗")]
            ))
            return
        st["lv"]   = lv
        st["step"] = "ask_hist"
        qr_items = [
            ("0回", "HIST:0"), ("1回", "HIST:1"), ("2回", "HIST:2"),
            ("3回以上", "HIST:3"), ("縮毛", "HIST:S"), ("パーマ", "HIST:P")
        ]
        qr = QuickReply(items=[
            QuickReplyItem(action=MessageAction(label=l, text=t))
            for l, t in qr_items
        ])
        api.reply_message(ReplyMessageRequest(
            reply_token=ev.reply_token,
            messages=[TextMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=qr)]
        ))
        return

    # ---- 履歴受信後にレシピ算出 ----
    if txt.startswith("HIST:") and st.get("step") == "ask_hist":
        hist = txt.split(":")[1]; lv = st["lv"]
        lab  = extract_lab(st["img"])  # ★今後：lab を使う場合
        # スコア計算（一例）
        df["score"] = (df["L"] - lv * 12).abs() * 0.5 + \
                      (df["formula"].str.contains("6%") & (hist == "S")) * 10
        top3 = df.nsmallest(3, "score")

        car = {"type": "carousel",
               "contents": [bubble_dict(r) for r in top3.itertuples()]}

        api.reply_message(ReplyMessageRequest(
            reply_token=ev.reply_token,
            messages=[FlexMessage(alt_text="おすすめレシピ", contents=car)]
        ))
        state.pop(uid, None)

# ─────────── health check (GET) ───────────
@app.route("/callback", methods=["GET"])
def health(): return "OK", 200

# ─────────── local run ───────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
