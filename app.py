# === app.py =========================================================
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookParser
from linebot.models import (
    MessageEvent, ImageMessage, TextMessage, TextSendMessage,
    QuickReply, QuickReplyButton, MessageAction
)
from dotenv import load_dotenv
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import openai

# ---------- env 読み込み ----------
load_dotenv()
CHANNEL_SECRET       = os.getenv("CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
openai.api_key       = os.getenv("OPENAI_API_KEY")

print("Loaded env:", CHANNEL_SECRET[:5] if CHANNEL_SECRET else "None",
      openai.api_key[:5] if openai.api_key else "None")

# ---------- LINE 初期化 ----------
app          = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
parser       = WebhookParser(CHANNEL_SECRET)

# ---------- kNN 準備 ----------
df  = pd.read_csv("recipes.csv")
X   = df[["L", "a", "b"]].values
knn = NearestNeighbors(n_neighbors=3).fit(X)

user_state = defaultdict(dict)   # user_id → {step, img, lv}

def extract_lab(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1,3).mean(axis=0)
    return lab

# ================= callback =================
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
            content = line_bot_api.get_message_content(ev.message.id)
            uid = ev.source.user_id
            user_state[uid] = {"step":"ask_lv", "img":content.content}

            line_bot_api.reply_message(
                ev.reply_token,
                TextSendMessage(text="現在の明度を 0〜19 の数字で送ってください📩")
            )
            print("📷 img recv:", uid)
            return "OK", 200

        # ---------- ② テキスト ----------
        if isinstance(ev.message, TextMessage):
            text = ev.message.text
            uid  = ev.source.user_id
            st   = user_state.get(uid, {})

            # --- LV 受付 ---
            if st.get("step")=="ask_lv":
                try:
                    lv = int(text)
                    if not 0<=lv<=19: raise ValueError
                except ValueError:
                    line_bot_api.reply_message(
                        ev.reply_token,
                        TextSendMessage(text="0〜19 の数字で送ってください❗")
                    )
                    return "OK",200

                st["lv"]=lv; st["step"]="ask_hist"
                quick = QuickReply(items=[
                    QuickReplyButton(action=MessageAction(label="ブリーチなし", text="HIST:0")),
                    QuickReplyButton(action=MessageAction(label="ブリーチ1回", text="HIST:1")),
                    QuickReplyButton(action=MessageAction(label="ブリーチ2回", text="HIST:2")),
                    QuickReplyButton(action=MessageAction(label="縮毛あり",   text="HIST:S")),
                ])
                line_bot_api.reply_message(
                    ev.reply_token,
                    TextSendMessage(text="ブリーチや縮毛の履歴は？", quick_reply=quick)
                )
                print("📝 lv recv:", lv)
                return "OK",200

            # --- 履歴受付 & GPT ---
            if text.startswith("HIST:") and st.get("step")=="ask_hist":
                hist = text.split(":")[1]
                lv   = st["lv"]; lab = extract_lab(st["img"])

                df["lv_diff"]  = (df["L"]-lv*12).abs()
                df["hist_pen"] = (df["formula"].str.contains("6%")&(hist=="S")).astype(int)
                df["score"]    = df["lv_diff"]*0.5 + df["hist_pen"]*10
                top3 = df.nsmallest(3,"score")

                replies=[]
                for r in top3.itertuples():
                    prompt=(f"以下のヘアカラー処方を美容師らしく一言で解説して。\n"
                            f"処方: {r.formula}\n40文字以内、日本語。")
                    try:
                        rsp=openai.ChatCompletion.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=60,temperature=0.7
                        )
                        comment=rsp.choices[0].message.content.strip()

                    except Exception as e:
                        print("GPT Error:",type(e).__name__,"-",e,file=sys.stderr)
                        traceback.print_exc()
                        comment="(解説取得エラー)"

                    replies.append(f"{r.name}\n{r.formula}\n{comment}")

                line_bot_api.reply_message(
                    ev.reply_token,
                    TextSendMessage(text="\n\n".join(replies))
                )
                print("✅ done:", uid)
                user_state.pop(uid,None)
                return "OK",200

    return "OK",200   # fallback

# ---------- Verify 用 GET ----------
@app.route("/callback", methods=["GET"])
def health_check():
    return "OK",200

# ---------- 起動 ----------
if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)
# ================================================================
