# === app.py  (Flex カルーセル + QuickReply ボタン) =========================
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookParser
from linebot.models import (
    MessageEvent, ImageMessage, TextMessage,
    TextSendMessage, QuickReply, QuickReplyButton, MessageAction,
    FlexSendMessage
)
from dotenv import load_dotenv
from openai import OpenAI
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# ---------- env ----------
load_dotenv()
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")

# ★ あなたの Static サイト URL
BASE_IMG = "https://aic-olorbot-static.onrender.com"

client  = OpenAI(api_key=OPENAI_KEY)
app     = Flask(__name__)
bot     = LineBotApi(CHANNEL_TOKEN)
parser  = WebhookParser(CHANNEL_SECRET)

# ---------- kNN ----------
df  = pd.read_csv("recipes.csv")
knn = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)

state = defaultdict(dict)          # user_id → {"step", "img", "lv"}

def extract_lab(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(axis=0)

# ======================================================================
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
            state[uid] = {"step":"ask_lv", "img":content.content}

            bot.reply_message(
                ev.reply_token,
                TextSendMessage(text="現在の明度を 0〜19 の数字で送ってください📩")
            )
            return "OK", 200

        # ---------- ② テキスト ----------
        if isinstance(ev.message, TextMessage):
            text = ev.message.text
            uid  = ev.source.user_id
            st   = state.get(uid, {})

            # ---- LV 受付 ----
            if st.get("step") == "ask_lv":
                try:
                    lv = int(text)
                    assert 0 <= lv <= 19
                except Exception:
                    bot.reply_message(
                        ev.reply_token,
                        TextSendMessage(text="0〜19 の数字で送ってね❗")
                    )
                    return "OK", 200

                st["lv"]   = lv
                st["step"] = "ask_hist"

                qr = QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="ブリーチなし",   text="HIST:0")),
                    QuickReplyButton(action=MessageAction(label="1回",          text="HIST:1")),
                    QuickReplyButton(action=MessageAction(label="2回",          text="HIST:2")),
                    QuickReplyButton(action=MessageAction(label="縮毛",         text="HIST:S")),
                ])
                bot.reply_message(
                    ev.reply_token,
                    TextSendMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=qr)
                )
                return "OK", 200

            # ---- 履歴を受信したら推論＋GPT 解説 ----
            if text.startswith("HIST:") and st.get("step") == "ask_hist":
                hist = text.split(":")[1]
                lv   = st["lv"]

                # kNN スコア
                df["score"] = (df["L"]-lv*12).abs()*0.5 + \
                              (df["formula"].str.contains("6%")&(hist=="S"))*10
                top3 = df.nsmallest(3, "score")

                bubbles = []
                for r in top3.itertuples():
                    prompt = (f"以下のヘアカラー処方を美容師らしく一言で解説して。\n"
                              f"処方: {r.formula}\n40文字以内、日本語。")
                    try:
                        rsp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=60,
                            temperature=0.7
                        )
                        comment = rsp.choices[0].message.content.strip()
                    except Exception as e:
                        traceback.print_exc()
                        comment = "(解説取得エラー)"

                    bubbles.append({
                        "type":"bubble",
                        "hero":{
                            "type":"image",
                            "url":f"{BASE_IMG}/{r.Name}.png",
                            "size":"full","aspectRatio":"1:1","aspectMode":"cover"
                        },
                        "body":{
                            "type":"box","layout":"vertical","spacing":"sm",
                            "contents":[
                                {"type":"text","text":r.Name,"weight":"bold","size":"md"},
                                {"type":"text","text":r.formula,"wrap":True,"size":"sm"},
                                {"type":"text","text":comment,"wrap":True,"size":"sm","color":"#888888"}
                            ]
                        }
                    })

                flex = FlexSendMessage(
                    alt_text="おすすめ処方",
                    contents={"type":"carousel","contents":bubbles}
                )
                bot.reply_message(ev.reply_token, flex)
                state.pop(uid, None)
                return "OK", 200

    return "OK", 200

# ---- Verify 用 ----
@app.route("/callback", methods=["GET"])
def health(): return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
# ===================================================================
