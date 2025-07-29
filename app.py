from flask import Flask, request, abort
from dotenv import load_dotenv
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from openai import OpenAI

# LINE SDK v3 modules
from linebot.v3.messaging import (
    MessagingApi, Configuration,
    ReplyMessageRequest, TextMessage,
    QuickReply, QuickReplyItem, MessageAction,
    FlexMessage, FlexContainer
)
from linebot.v3.messaging.models import (
    TextMessageContent, ImageMessageContent,
    Bubble, ImageComponent, BoxComponent, TextComponent
)
from linebot.v3.webhooks import WebhookParser
from linebot.v3.webhooks.models import MessageEvent

# ---------- config & init ----------
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
parser = WebhookParser(channel_secret=os.getenv("CHANNEL_SECRET"))
cfg = Configuration(access_token=os.getenv("CHANNEL_ACCESS_TOKEN"))
api = MessagingApi(cfg)
CHIP_BASE = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

df = pd.read_csv("recipes.csv")
knn = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)
state = defaultdict(dict)

# ---------- helper ----------
def extract_lab(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
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
        print("GPT Error:", e)
        return "(解説取得エラー)"

def make_bubble(rec) -> Bubble:
    return Bubble(
        hero=ImageComponent(
            url=f"{CHIP_BASE}/{rec.Name}.png",
            size="full", aspect_mode="cover", aspect_ratio="1:1"
        ),
        body=BoxComponent(
            layout="vertical",
            spacing="sm",
            contents=[
                TextComponent(text=rec.Name, weight="bold", size="md"),
                TextComponent(text=rec.formula, wrap=True, size="sm"),
                TextComponent(text=gpt_comment(rec.formula),
                              wrap=True, size="sm", color="#888")
            ]
        )
    )

# ========== Webhook ==========
@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        events = parser.parse(body, sig)
    except Exception:
        abort(400)

    for ev in events:
        # 画像
        if isinstance(ev, MessageEvent) and isinstance(ev.message, ImageMessageContent):
            uid = ev.source.user_id
            content = api.get_message_content(ev.message.id)
            state[uid] = {"step": "ask_lv", "img": content}
            msg = TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")
            api.reply_message(ReplyMessageRequest(reply_token=ev.reply_token, messages=[msg]))
            return "OK", 200

        # テキスト
        if isinstance(ev, MessageEvent) and isinstance(ev.message, TextMessageContent):
            txt = ev.message.text.strip()
            uid = ev.source.user_id
            st = state.get(uid, {})

            if st.get("step") == "ask_lv":
                try:
                    lv = int(txt); assert 0 <= lv <= 19
                except:
                    msg = TextMessage(text="0〜19 の数字で送ってね❗")
                    api.reply_message(ReplyMessageRequest(reply_token=ev.reply_token, messages=[msg]))
                    return "OK", 200

                st["lv"] = lv
                st["step"] = "ask_hist"

                qr = QuickReply(items=[
                    QuickReplyItem(action=MessageAction(label="0回", text="HIST:0")),
                    QuickReplyItem(action=MessageAction(label="1回", text="HIST:1")),
                    QuickReplyItem(action=MessageAction(label="2回", text="HIST:2")),
                    QuickReplyItem(action=MessageAction(label="3回以上", text="HIST:3")),
                    QuickReplyItem(action=MessageAction(label="縮毛", text="HIST:S")),
                    QuickReplyItem(action=MessageAction(label="パーマ", text="HIST:P")),
                ])
                msg = TextMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=qr)
                api.reply_message(ReplyMessageRequest(reply_token=ev.reply_token, messages=[msg]))
                return "OK", 200

            if txt.startswith("HIST:") and st.get("step") == "ask_hist":
                hist = txt.split(":")[1]
                lv   = st["lv"]
                lab  = extract_lab(st["img"])
                df["score"] = (df["L"] - lv * 12).abs() * 0.5 + \
                              (df["formula"].str.contains("6%") & (hist == "S")) * 10
                top3 = df.nsmallest(3, "score")

                flex = FlexContainer(type="carousel", contents=[make_bubble(r) for r in top3.itertuples()])
                msg = FlexMessage(alt_text="おすすめレシピ", contents=flex)
                api.reply_message(ReplyMessageRequest(reply_token=ev.reply_token, messages=[msg]))
                state.pop(uid, None)
                return "OK", 200

    return "OK", 200

@app.route("/callback", methods=["GET"])
def health(): return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
