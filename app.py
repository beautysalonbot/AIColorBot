# === imports (LINE SDK v3.4.0 & others) ======================================
from flask import Flask, request, abort
from dotenv import load_dotenv
from openai import OpenAI
from linebot.v3.messaging import (
    Configuration, MessagingApi,
    ReplyMessageRequest, TextMessage, FlexMessage,
    QuickReply, QuickReplyItem, MessageAction
)
from linebot.v3.messaging.models import (
    Image, Box, Text, Bubble, Carousel
)
from linebot.v3.webhooks import WebhookParser
from linebot.v3.webhooks.models import (
    MessageEvent, TextMessageContent, ImageMessageContent
)
# ----------------------------------------------------------------------------
import os, sys, traceback, cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# === init ===============================================================
load_dotenv()
CHAN_SECRET = os.getenv("CHANNEL_SECRET")
CHAN_TOKEN  = os.getenv("CHANNEL_ACCESS_TOKEN")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY")
CHIP_BASE   = os.getenv("CHIP_BASE", "https://aic-olorbot-static.onrender.com")

client = OpenAI(api_key=OPENAI_KEY)
cfg    = Configuration(access_token=CHAN_TOKEN)
api    = MessagingApi(cfg)
app    = Flask(__name__)
parser = WebhookParser(CHAN_SECRET)

# === kNN 準備 ===========================================================
df  = pd.read_csv("recipes.csv")  # Name,L,a,b,formula,...
knn = NearestNeighbors(n_neighbors=3).fit(df[["L", "a", "b"]].values)
state = defaultdict(dict)

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
            max_tokens
