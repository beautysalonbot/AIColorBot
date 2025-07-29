# === imports (SDK v3) ==========================================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI
from linebot.v3.messaging import (
    MessagingApi, Configuration,
    ReplyMessageRequest,
    TextMessage, QuickReply, QuickReplyItem, MessageAction,
    FlexMessage, FlexBubble, FlexCarousel,
)
from linebot.v3.messaging.models import (
    TextMessageContent, ImageMessageContent,
    Carousel, Bubble, ImageComponent, BoxComponent, TextComponent
)
from linebot.v3.webhooks import WebhookHandler
# ---------------------------------------------------------------
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# === config & init =============================================
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

client = OpenAI(api_key=OPENAI_KEY)
cfg    = Configuration(access_token=CHAN_TOKEN)
api    = MessagingApi(cfg)
app    = Flask(__name__)
handler = WebhookHandler(CHAN_SECRET)

# === k-NN 下ごしらえ ==========================================
df = pd.read_csv("recipes.csv")             # Name,L,a,b,formula …
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)
state = defaultdict(dict)                   # user_id → {step,img,lv}

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
        print("GPT Error:", type(e).__name__, "-", e, file=sys.stderr)
        return "(解説取得エラー)"

def bubble(rec) -> Bubble:
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
                TextComponent(text=gpt_comment(rec.formula), wrap=True, size="sm", color="#888"),
            ]
        )
    )

# === Webhook ===================================================
@app.route("/callback", methods=["POST"])
def callback() -> tuple[str, int]:
    sig  = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        events = handler.parse(body, sig)
    except Exception as e:
        print("Webhook parse error:", e, file=sys.stderr)
        abort(400)

    for ev in events:
        if isinstance(ev.message, ImageMessageContent):
            uid = ev.source.user_id
            img_bytes = api.get_message_content(ev.message.id).body
            state[uid] = {"step": "ask_lv", "img": img_bytes}

            api.reply_message(
                ReplyMessageRequest(
                    reply_token=ev.reply_token,
                    messages=[TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")]
                )
            )
            return "OK", 200

        if isinstance(ev.message, TextMessageContent):
            txt = ev.message.text.strip()
            uid = ev.source.user_id
            st  = state.get(uid, {})

            if st.get("step") == "ask_lv":
                try:
                    lv = int(txt); assert 0 <= lv <= 19
                except:
                    api.reply_message(
                        ReplyMessageRequest(
                            reply_token=ev.reply_token,
                            messages=[TextMessage(text="0〜19 の数字で送ってね❗")]
                        )
                    )
                    return "OK", 200

                st["lv"]   = lv
                st["step"] = "ask_hist"

                qr = QuickReply(items=[
                    QuickReplyItem(action=MessageAction(label="0回", text="HIST:0")),
                    QuickReplyItem(action=MessageAction(label="1回", text="HIST:1")),
                    QuickReplyItem(action=MessageAction(label="2回", text="HIST:2")),
                    QuickReplyItem(action=MessageAction(label="3回以上", text="HIST:3")),
                    QuickReplyItem(action=MessageAction(label="縮毛", text="HIST:S")),
                    QuickReplyItem(action=MessageAction(label="パーマ", text="HIST:P")),
                ])
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=ev.reply_token,
                        messages=[TextMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=qr)]
                    )
                )
                return "OK", 200

            if txt.startswith("HIST:") and st.get("step") == "ask_hist":
                hist = txt.split(":")[1]
                lv   = st["lv"]
                lab  = extract_lab(st["img"])

                df["score"] = (df["L"]-lv*12).abs()*0.5 + \
                              (df["formula"].str.contains("6%") & (hist=="S"))*10
                top3 = df.nsmallest(3, "score")

                car = Carousel(contents=[bubble(r) for r in top3.itertuples()])
                api.reply_message(
                    ReplyMessageRequest(
                        reply_token=ev.reply_token,
                        messages=[FlexMessage(alt_text="おすすめレシピ", contents=car)]
                    )
                )
                state.pop(uid, None)
                return "OK", 200

    return "OK", 200

# === GET for healthcheck =======================================
@app.route("/callback", methods=["GET"])
def health(): return "OK", 200

# === local run =================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
