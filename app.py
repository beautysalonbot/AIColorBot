from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI
from linebot.v3.messaging import (
    Configuration, MessagingApi, ReplyMessageRequest,
    TextMessage, FlexMessage, QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.webhooks import (
    CallbackRequest, MessageEvent, TextMessageContent, ImageMessageContent
)
from linebot.v3.webhooks.models import (
    MessageEvent as EventMessageEvent,
    TextMessageContent as EventTextMessageContent,
    ImageMessageContent as EventImageMessageContent,
)
from linebot.v3.messaging.models import (
    FlexContainer, BubbleContainer, CarouselContainer,
    ImageComponent, BoxComponent, TextComponent
)

import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import base64

# === init ===
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

client = OpenAI(api_key=OPENAI_KEY)
cfg    = Configuration(access_token=CHAN_TOKEN)
api    = MessagingApi(cfg)
app    = Flask(__name__)
state  = defaultdict(dict)

# === kNN 準備 ===
df  = pd.read_csv("recipes.csv")  # Name,L,a,b,formula,...
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)

def extract_lab(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return lab.mean(0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく一言で解説して。\n処方: {formula}\n40文字以内、日本語。"
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

@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_data(as_text=True)
    try:
        events = CallbackRequest.from_json(body).events
        for event in events:
            if isinstance(event, EventMessageEvent):
                handle_event(event)
    except Exception as e:
        print("Callback Error:", e, file=sys.stderr)
    return "OK", 200

def handle_event(event: EventMessageEvent):
    msg = event.message
    uid = event.source.user_id
    if isinstance(msg, EventTextMessageContent):
        text = msg.text.lower()
        if "リセット" in text:
            state[uid].clear()
            reply(uid, "状態をリセットしました。")
        else:
            reply(uid, "画像を送ってください📷")
    elif isinstance(msg, EventImageMessageContent):
        try:
            blob = api.get_message_content(msg.id)
            b = b''.join(chunk for chunk in blob.iter_content(None))
            lab = extract_lab(b)
            dists, indices = knn.kneighbors([lab])
            bubbles = []
            for i in indices[0]:
                row = df.iloc[i]
                comment = gpt_comment(row["formula"])
                bubbles.append(BubbleContainer(
                    hero=ImageComponent(
                        url=f"{CHIP_BASE}/chips/{row['Name']}.jpg",
                        size="full", aspectRatio="1:1", aspectMode="cover"
                    ),
                    body=BoxComponent(
                        layout="vertical",
                        contents=[
                            TextComponent(text=row["Name"], weight="bold", size="xl"),
                            TextComponent(text=comment, wrap=True, margin="md", size="sm")
                        ]
                    )
                ))
            carousel = CarouselContainer(contents=bubbles)
            msg = FlexMessage(alt_text="おすすめレシピ", contents=carousel)
            api.reply_message(ReplyMessageRequest(reply_token=event.reply_token, messages=[msg]))
        except Exception as e:
            traceback.print_exc()
            reply(uid, "画像処理でエラーが起きました💥")

def reply(user_id: str, text: str):
    msg = TextMessage(text=text)
    api.reply_message(ReplyMessageRequest(reply_token=user_id, messages=[msg]))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
