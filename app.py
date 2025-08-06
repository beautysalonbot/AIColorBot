# === imports ================================================================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI

from linebot.v3 import WebhookHandler, WebhookParser        # ★ここが正解
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient,
    MessagingApi,          # 返信など
    MessagingApiBlob,      # ★ BLOB 用
    ReplyMessageRequest, TextMessage, FlexMessage,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.webhooks import (
    MessageEvent, TextMessageContent, ImageMessageContent
)

# ---- other libs ------------------------------------------------------------
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
# ============================================================================

# 1) 環境変数 ----------------------------------------------------------------
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE",
              "https://aic-olorbot-static.onrender.com")

# 2) SDK 初期化 --------------------------------------------------------------
handler       = WebhookHandler(CHAN_SECRET)
configuration = Configuration(access_token=CHAN_TOKEN)
api_client = ApiClient(configuration)
bot        = MessagingApi(api_client)         # メッセージ送信用
blob_api   = MessagingApiBlob(api_client)     # ★ BLOB（画像ダウンロード）用
client = OpenAI(api_key=OPENAI_KEY)
app    = Flask(__name__)

# 3) k-NN 下準備 -------------------------------------------------------------
df    = pd.read_csv("recipes.csv")            # Name,L,a,b,formula,...
knn   = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)
state = defaultdict(dict)                     # user_id → {step, lv, img}

# 4) ユーティリティ -----------------------------------------------------------
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
    """Flex Bubble を純 JSON で作成（追加 import 不要）"""
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
                {"type": "text", "text": rec.Name,    "weight": "bold", "size": "md"},
                {"type": "text", "text": rec.formula, "wrap": True,     "size": "sm"},
                {"type": "text", "text": gpt_comment(rec.formula),
                 "wrap": True, "size": "sm", "color": "#888"}
            ]
        }
    }

# === Webhookエンドポイント ===================================================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body      = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)       # ← これが正式ルート
    except InvalidSignatureError:
        print("[Signature Error] channel secret/token 不一致", file=sys.stderr)
        abort(400)
    except Exception as e:
        print("[Webhook Error]", e, file=sys.stderr)
        abort(400)

    return "OK", 200

# === ハンドラ群 ==============================================================
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event: MessageEvent):
    uid = event.source.user_id
    # v3.18 ではストリームが返るので read() で bytes 化
    raw = blob_api.get_message_content(event.message.id).read()
    state[uid] = {"step": "ask_lv", "img": raw}

    bot.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[TextMessage(text="現在の明度を 0〜19 の数字で送ってください📩")]
        )
    )

@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event: MessageEvent):
    uid  = event.source.user_id
    text = event.message.text.strip()
    st   = state.get(uid, {})

    # --- 明度入力フェーズ ----------------------------------------------------
    if st.get("step") == "ask_lv":
        try:
            lv = int(text); assert 0 <= lv <= 19
        except Exception:
            bot.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="0〜19 の数字で送ってね❗")]
                )
            )
            return

        st["lv"]   = lv
        st["step"] = "ask_hist"

        qr_items = [
            ("0回", "HIST:0"), ("1回", "HIST:1"), ("2回", "HIST:2"),
            ("3回以上", "HIST:3"), ("縮毛", "HIST:S"), ("パーマ", "HIST:P")
        ]
        quick = QuickReply(items=[
            QuickReplyItem(action=MessageAction(label=l, text=t))
            for l, t in qr_items
        ])

        bot.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="ブリーチ・縮毛などの履歴を選んでね", quick_reply=quick)]
            )
        )
        return

    # --- 履歴受領後のおすすめ ------------------------------------------------
    if text.startswith("HIST:") and st.get("step") == "ask_hist":
        hist = text.split(":")[1]
        lv   = st["lv"]
        lab  = extract_lab(st["img"])          # lab は今後に活用予定

        # スコア計算例
        df["score"] = (df["L"] - lv * 12).abs() * 0.5 + \
                      (df["formula"].str.contains("6%") & (hist == "S")) * 10
        top3 = df.nsmallest(3, "score")

        carousel = {
            "type": "carousel",
            "contents": [bubble_dict(r) for r in top3.itertuples()]
        }

        bot.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[FlexMessage(alt_text="おすすめレシピ", contents=carousel)]
            )
        )
        state.pop(uid, None)

# === Health check (Render) ===================================================
@app.route("/callback", methods=["GET"])
def health():    # Render のヘルスチェック用
    return "OK", 200

# === Local run ===============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
