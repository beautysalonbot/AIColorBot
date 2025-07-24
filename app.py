# === app.py (Flex カルーセル対応・hist 入力修正版) =====================
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookParser
from linebot.models import (
    MessageEvent, ImageMessage, TextMessage, TextSendMessage,
    FlexSendMessage
)
from dotenv import load_dotenv
from openai import OpenAI
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# ---------- 環境 ----------
load_dotenv()
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")
CHANNEL_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY     = os.getenv("OPENAI_API_KEY")
CHIP_BASE      = os.getenv("CHIP_BASE",
                           "https://aic-olorbot-static.onrender.com")

client = OpenAI(api_key=OPENAI_KEY)
app    = Flask(__name__)
bot    = LineBotApi(CHANNEL_TOKEN)
parser = WebhookParser(CHANNEL_SECRET)

# ---------- データ ----------
df = pd.read_csv("recipes.csv")                         # name,L,a,b,formula
knn = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)
state = defaultdict(dict)                              # uid → step,img,lv

# ---------- util ----------
def extract_lab(b: bytes):
    img = cv2.imdecode(np.frombuffer(b,np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(axis=0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく一言で解説して。\n処方: {formula}\n40文字以内、日本語。"
    try:
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=60, temperature=0.7
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("GPT Error:",type(e).__name__,e,file=sys.stderr)
        traceback.print_exc()
        return "(解説取得エラー)"

def make_bubble(row, comment):
    return {
        "type":"bubble",
        "hero":{
            "type":"image",
            "url":f"{CHIP_BASE}/{row['name']}.png",
            "size":"full","aspectRatio":"1:1","aspectMode":"cover"
        },
        "body":{
            "type":"box","layout":"vertical","spacing":"sm",
            "contents":[
                {"type":"text","text":row["name"],"weight":"bold","size":"md"},
                {"type":"text","text":row["formula"],"size":"sm","wrap":True},
                {"type":"text","text":comment,"size":"sm","wrap":True,"color":"#888888"}
            ]
        }
    }

# ====================================================================
@app.route("/callback",methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        events = parser.parse(body,sig)
    except Exception: abort(400)

    for ev in events:
        # ---------- 画像 ----------
        if isinstance(ev,MessageEvent) and isinstance(ev.message,ImageMessage):
            uid  = ev.source.user_id
            img  = bot.get_message_content(ev.message.id).content
            state[uid]={"step":"ask_lv","img":img}
            bot.reply_message(ev.reply_token,
                TextSendMessage(text="現在の明度を 0〜19 の数字で送ってください📩"))
            continue

        # ---------- テキスト ----------
        if isinstance(ev,MessageEvent) and isinstance(ev.message,TextMessage):
            uid,text = ev.source.user_id, ev.message.text.strip()
            st = state.get(uid,{})

            # LV 受付
            if st.get("step")=="ask_lv":
                try:
                    lv=int(text); assert 0<=lv<=19
                except Exception:
                    bot.reply_message(ev.reply_token,
                        TextSendMessage(text="0〜19 の数字で送ってね❗"))
                    continue
                st["lv"]=lv; st["step"]="ask_hist"
                bot.reply_message(ev.reply_token,
                    TextSendMessage(text="ブリーチ履歴を 0/1/2/S で返信してね\n(0=なし,S=縮毛)"))
                continue

            # 履歴受付（0/1/2/S いずれか）
            if st.get("step")=="ask_hist":
                hist=text.upper()
                if hist not in {"0","1","2","S"}:
                    bot.reply_message(ev.reply_token,
                        TextSendMessage(text="0/1/2/S で送ってね❗"))
                    continue

                lv   = st["lv"]
                lab  = extract_lab(st["img"])

                # スコア計算
                df["score"]  = (df["L"]-lv*12).abs()*0.5 \
                             + (df["formula"].str.contains("6%")&(hist=="S"))*10
                top3=df.nsmallest(3,"score")

                bubbles=[make_bubble(r,gpt_comment(r["formula"]))
                         for _,r in top3.iterrows()]

                flex=FlexSendMessage(
                    alt_text="おすすめレシピ",
                    contents={"type":"carousel","contents":bubbles}
                )
                bot.reply_message(ev.reply_token,flex)
                state.pop(uid,None)
                continue
    return "OK",200

# ---------- ヘルスチェック ----------
@app.route("/callback",methods=["GET"])
def health(): return "OK",200

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)))
# ====================================================================
