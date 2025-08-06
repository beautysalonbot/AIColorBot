# === imports ================================================================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI

from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient,
    MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, TextMessage, FlexMessage,
    QuickReply, QuickReplyItem, MessageAction,
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, ImageMessageContent
)

# ---- other libs ------------------------------------------------------------
import os, sys, traceback
from collections import defaultdict

import cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
# ============================================================================

# 1) env ---------------------------------------------------------------------
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

# 2) LINE SDK ---------------------------------------------------------------
configuration = Configuration(access_token=CHAN_TOKEN)
handler = WebhookHandler(CHAN_SECRET)

# 3) OpenAI ------------------------------------------------------------------
openai_client = OpenAI(api_key=OPENAI_KEY)

# 4) Flask -------------------------------------------------------------------
app = Flask(__name__)

# 5) k-NN --------------------------------------------------------------------
df  = pd.read_csv("recipes.csv")                # Name,L,a,b,formula,...
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)

state: dict[str, dict] = defaultdict(dict)      # user_id → {step, lv, img}
# 6) helpers -----------------------------------------------------------------
def extract_lab(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)

def gpt_comment(formula: str) -> str:
    prompt = f"以下のヘアカラー処方を美容師らしく40文字以内で解説して。\n処方: {formula}"
    try:
        rsp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60, temperature=0.7,
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("[GPT Error]", e, file=sys.stderr)
        traceback.print_exc()
        return "(解説取得エラー)"

def bubble_dict(rec) -> dict:
    return {
        "type": "bubble",
        "hero": {"type": "image", "url": f"{CHIP_BASE}/{rec.Name}.png",
                 "size": "full", "aspectMode": "cover", "aspectRatio": "1:1"},
        "body": {"type": "box", "layout": "vertical", "spacing": "sm",
                 "contents": [
                     {"type": "text", "text": rec.Name, "weight": "bold", "size": "md"},
                     {"type": "text", "text": rec.formula, "wrap": True, "size": "sm"},
                     {"type": "text", "text": gpt_comment(rec.formula),
                      "wrap": True, "size": "sm", "color": "#888"},
                 ]},
    }

# === Webhook entry ==========================================================
@app.route("/callback", methods=["POST"])
def callback() -> tuple[str, int]:
    signature = request.headers.get("X-Line-Signature", "")
    body      = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "[Signature Error]")
    except Exception as e:
        print("[Webhook Error]", e, file=sys.stderr)
        abort(400)
    return "OK", 200

# === handlers ==============================================================-
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent) -> None:
    uid = event.source.user_id
    # ApiClient は都度生成が安全
    with ApiClient(configuration) as c:
        raw_resp = MessagingApiBlob(c).get_message_content(event.message.id)
    state[uid] = {"step": "ask_lv", "img": raw_resp.body}

    with ApiClient(configuration) as c:
        MessagingApi(c).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")]
            )
        )

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event: MessageEvent) -> None:
    uid  = event.source.user_id
    text = event.message.text.strip()
    st   = state.get(uid, {})

    # ---- 明度入力 ----------------------------------------------------------
    if st.get("step") == "ask_lv":
        try:
            lv = int(text); assert 0 <= lv <= 19
        except Exception:
            with ApiClient(configuration) as c:
                MessagingApi(c).reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text="0〜19 の数字で送ってね❗")]
                    )
                )
            return

        st.update({"lv": lv, "step": "ask_hist"})
        quick = QuickReply(items=[
            QuickReplyItem(action=MessageAction(label=l, text=t))
            for l, t in (("0回","HIST:0"),("1回","HIST:1"),("2回","HIST:2"),
                         ("3回以上","HIST:3"),("縮毛","HIST:S"),("パーマ","HIST:P"))
        ])

        with ApiClient(configuration) as c:
            MessagingApi(c).reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(
                        text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=quick)]
                )
            )
        return

    # ---- 履歴入力後 --------------------------------------------------------
    if text.startswith("HIST:") and st.get("step") == "ask_hist":
        hist = text.split(":", 1)[1]
        lv   = st["lv"];  lab = extract_lab(st["img"])

        tmp = df.copy()
        tmp["score"] = (tmp["L"] - lv * 12).abs() * 0.5 + \
                       ((tmp["formula"].str.contains("6%")) & (hist == "S")) * 10
        top3 = tmp.nsmallest(3, "score")

        carousel = {"type": "carousel",
                    "contents": [bubble_dict(r) for r in top3.itertuples()]}

        with ApiClient(configuration) as c:
            MessagingApi(c).reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[FlexMessage(alt_text="おすすめレシピ", contents=carousel)]
                )
            )
        state.pop(uid, None)

# === Health check ===========================================================
@app.route("/health", methods=["GET"])
def health() -> tuple[str, int]:
    return "OK", 200

# === local run ============================================================== 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
