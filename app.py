# === app.py  (LINE SDK v3  + Flex Carousel) ==============================
from __future__ import annotations
import os, sys, traceback, cv2, numpy as np, pandas as pd
from collections import defaultdict
from flask import Flask, request, abort
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

# ---------- LINE v3 SDK ----------
from linebot.v3.webhook import WebhookParser
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, TextMessage, FlexMessage
)

# ---------- 環境変数 ----------
load_dotenv()
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
ACCESS_TOKEN   = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")
CHIP_BASE      = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

# ---------- LINE クライアント ----------
conf        = Configuration(access_token=ACCESS_TOKEN)
api_client  = ApiClient(conf)
msg_api     = MessagingApi(api_client)
blob_api    = MessagingApiBlob(api_client)
parser      = WebhookParser(CHANNEL_SECRET)

# ---------- OpenAI ----------
client = OpenAI(api_key=OPENAI_KEY)

# ---------- レシピ k‑NN ----------
df   = pd.read_csv("recipes.csv")
knn  = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)

user_state: dict[str,dict] = defaultdict(dict)   # user_id → {step,img,lv}

def extract_lab(img_bytes: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(axis=0)

def make_bubble(rec, comment: str) -> dict:
    return {
        "type":"bubble",
        "hero":{
            "type":"image",
            "url":f"{CHIP_BASE}/{rec.Name}.png",
            "size":"full","aspectRatio":"1:1","aspectMode":"cover"
        },
        "body":{
            "type":"box","layout":"vertical","spacing":"sm",
            "contents":[
                {"type":"text","text":rec.Name,"weight":"bold","size":"md"},
                {"type":"text","text":rec.formula,"wrap":True,"size":"sm"},
                {"type":"text","text":comment,"wrap":True,"size":"sm","color":"#888"}
            ]
        }
    }

# ---------- Flask ----------
app = Flask(__name__)

@app.route("/callback", methods=["POST"])
def callback() -> tuple[str,int]:
    sig  = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        events = parser.parse(body, sig)
    except Exception:
        abort(400)

    for ev in events:
        # ① 画像を受信
        if ev.message.type == "image":
            uid = ev.source.user_id
            content = blob_api.get_message_content(message_id=ev.message.id)
            user_state[uid] = {"step":"ask_lv","img":content.body}
            _reply(ev.reply_token, "現在の明度を 0〜19 の数字で送ってください📩")
            return "OK",200

        # ② テキストを受信
        if ev.message.type == "text":
            uid  = ev.source.user_id
            text = ev.message.text.strip()
            st   = user_state.get(uid,{})

            # ---- 明度ステップ ----
            if st.get("step")=="ask_lv":
                try:
                    lv=int(text); assert 0<=lv<=19
                except Exception:
                    _reply(ev.reply_token,"0〜19 の数字で送ってね❗")
                    return "OK",200
                st.update(lv=lv, step="ask_hist")
                _reply(ev.reply_token,"ブリーチ履歴を 0/1/2/S で返信してね (S=縮毛)")
                return "OK",200

            # ---- 履歴ステップ ----
            if text.startswith("HIST:") and st.get("step")=="ask_hist":
                hist=text.split(":")[1]; lv=st["lv"]; lab=extract_lab(st["img"])
                df["score"]=(df["L"]-lv*12).abs()*0.5 + (df["formula"].str.contains("6%")&(hist=="S"))*10
                top3=df.nsmallest(3,"score")

                bubbles=[]
                for r in top3.itertuples():
                    # GPT で 40文字解説
                    try:
                        rsp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user",
                                       "content":f"以下のヘアカラー処方を美容師らしく一言で解説して。40文字以内、日本語。\n処方: {r.formula}"}],
                            max_tokens=60,temperature=0.7)
                        comment=rsp.choices[0].message.content.strip()
                    except Exception as e:
                        traceback.print_exc()
                        comment="(解説取得エラー)"
                    bubbles.append(make_bubble(r,comment))

                flex = FlexMessage(alt_text="おすすめレシピ",
                                   contents={"type":"carousel","contents":bubbles})
                msg_api.reply_message(ReplyMessageRequest(
                    reply_token=ev.reply_token,
                    messages=[flex]
                ))
                user_state.pop(uid,None)
                return "OK",200

    return "OK",200

# ---- LINE Verify 用 GET ----
@app.route("/callback",methods=["GET"])
def health(): return "OK",200

def _reply(token:str, text:str):
    msg_api.reply_message(ReplyMessageRequest(
        reply_token=token,
        messages=[TextMessage(text=text)]
    ))

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
# ===================================================================
