# === app.py (Flex カルーセル対応) ===================================
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

# ---------- config ----------
load_dotenv()
CHANNEL_SECRET  = os.getenv("CHANNEL_SECRET")
CHANNEL_TOKEN   = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")
CHIP_BASE       = os.getenv("CHIP_BASE", "https://aicolor-chips.onrender.com")

client       = OpenAI(api_key=OPENAI_KEY)
app          = Flask(__name__)
bot          = LineBotApi(CHANNEL_TOKEN)
parser       = WebhookParser(CHANNEL_SECRET)

# ---------- kNN ----------
df  = pd.read_csv("recipes.csv")
knn = NearestNeighbors(n_neighbors=3).fit(df[["L","a","b"]].values)

state = defaultdict(dict)   # user_id → step/img/lv

def extract_lab(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(axis=0)

def make_card(row, comment):
    return {
        "type":"bubble",
        "hero":{
            "type":"image",
            "url": f"{CHIP_BASE}/{row['name']}.png",
            "size":"full",
            "aspectRatio":"1:1",
            "aspectMode":"cover"
        },
        "body":{
            "type":"box","layout":"vertical",
            "contents":[
                {"type":"text","text":row['name'],"weight":"bold","size":"lg"},
                {"type":"text","text":row['formula'],"wrap":True,"size":"sm","margin":"md"},
                {"type":"text","text":comment,"wrap":True,"size":"sm","color":"#999999","margin":"sm"}
            ]
        }
    }

# ================= callback =================
@app.route("/callback", methods=["POST"])
def callback():
    sig  = request.headers.get("X-Line-Signature","")
    body = request.get_data(as_text=True)
    try:
        events = parser.parse(body, sig)
    except Exception:
        abort(400)

    for ev in events:
        if isinstance(ev, MessageEvent) and isinstance(ev.message, ImageMessage):
            content = bot.get_message_content(ev.message.id)
            uid = ev.source.user_id
            state[uid]={"step":"ask_lv","img":content.content}
            bot.reply_message(ev.reply_token, TextSendMessage(text="現在の明度を 0〜19 の数字で送ってください📩"))
            return "OK",200

        if isinstance(ev.message, TextMessage):
            text, uid = ev.message.text, ev.source.user_id
            st = state.get(uid,{})

            if st.get("step")=="ask_lv":
                try:
                    lv=int(text); assert 0<=lv<=19
                except Exception:
                    bot.reply_message(ev.reply_token,TextSendMessage(text="0〜19の数字で送ってね❗"))
                    return "OK",200
                st["lv"]=lv; st["step"]="ask_hist"
                bot.reply_message(ev.reply_token,TextSendMessage(text="ブリーチ履歴を 0/1/2/S で返信してね\n(0=なし,S=縮毛)"))
                return "OK",200

            if text.startswith("HIST:") and st.get("step")=="ask_hist":
                hist=text.split(":")[1]; lv=st["lv"]; lab=extract_lab(st["img"])

                df["score"] = (df["L"]-lv*12).abs()*0.5 + (df["formula"].str.contains("6%")&(hist=="S"))*10
                top3 = df.nsmallest(3,"score")

                bubbles=[]
                for r in top3.itertuples():
                    prompt=f"以下のヘアカラー処方を美容師らしい一言で解説。40文字以内。\n処方: {r.formula}"
                    try:
                        rsp=client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=60,temperature=0.7
                        )
                        comment=rsp.choices[0].message.content.strip()
                    except Exception as e:
                        print("GPT Error:",type(e).__name__,"-",e,file=sys.stderr)
                        traceback.print_exc()
                        comment="(解説取得エラー)"
                    bubbles.append(make_card(r, comment))

                flex = FlexSendMessage(alt_text="おすすめレシピ", contents={"type":"carousel","contents":bubbles})
                bot.reply_message(ev.reply_token, flex)
                state.pop(uid,None)
                return "OK",200

    return "OK",200

@app.route("/callback",methods=["GET"])
def health(): return "OK",200

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
# ================================================================
