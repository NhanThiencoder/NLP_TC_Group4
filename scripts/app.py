import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import unicodedata
import requests
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import io

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="My Reading Trends",
    page_icon="eye",
    layout="wide"
)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
if not MODEL_DIR.exists(): MODEL_DIR = BASE_DIR

# --- 1. KHỞI TẠO SESSION STATE (BỘ NHỚ TẠM) ---
if 'history' not in st.session_state:
    st.session_state['history'] = []  # List chứa các bài đã phân tích

# --- 2. ĐỊNH NGHĨA MODEL & XỬ LÝ (GIỮ NGUYÊN TỪ TRƯỚC) ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)
        last_hidden = h_n[-1]; out = self.fc(last_hidden); return out

STOPWORD_PATH = BASE_DIR / "data" / "final" / "vietnamese-stopwords-dash.txt"

def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f.readlines()])
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file stopwords tại {filepath}")
        return {"thì", "là", "mà"}

STOPWORDS = load_stopwords(STOPWORD_PATH)

def normalize_text(text): return unicodedata.normalize('NFC', text)

def preprocess_text(text):
    text = normalize_text(text)
    tokenized = ViTokenizer.tokenize(text)
    words = tokenized.split()
    clean_words = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(clean_words)

def crawl_news_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Logic lấy nội dung chính (mở rộng thêm các báo khác ở đây)
        content = ""
        title = "Bài viết từ Link"
        
        # Tiêu đề
        if soup.title: title = soup.title.string
        
        # Nội dung (Thử các class phổ biến)
        paragraphs = soup.find_all('p', class_=['Normal', 'description', 'content'])
        if not paragraphs: paragraphs = soup.find_all('p') # Fallback lấy tất cả thẻ p
        
        content = "\n".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        
        if len(content) < 100: return None, None, "Nội dung quá ngắn hoặc không crawl được."
        return title, content, None
    except Exception as e: return None, None, str(e)

@st.cache_resource
def load_models():
    try:
        le = joblib.load(MODEL_DIR / "label_encoder.pkl")
        tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
        lr = joblib.load(MODEL_DIR / "logistic_regression.pkl")
        return le, tfidf, lr
    except: return None, None, None

le, tfidf, model = load_models()

# --- 3. GIAO DIỆN CHÍNH ---

# Sidebar: Nút Reset
with st.sidebar:
    st.title("⚙️ Cài đặt")
    if st.button("🗑️ Xóa lịch sử", type="primary"):
        st.session_state['history'] = []
        st.rerun()
    st.info("Hệ thống sẽ tích lũy các bài bạn nhập vào để phân tích xu hướng đọc.")

st.title("📊 Personal Content Analyzer")
st.markdown("Hệ thống phân tích xu hướng nội dung người dùng (User Profiling).")

if not le:
    st.error("❌ Thiếu model. Vui lòng kiểm tra folder 'models'."); st.stop()

# --- KHU VỰC 1: NHẬP LIỆU (ADD TO LIST) ---
with st.container(border=True):
    st.subheader("➕ Thêm nội dung mới")
    
    # Dùng tabs cho gọn
    tab_link, tab_text, tab_file = st.tabs(["🔗 Nhập Link", "📝 Nhập Văn bản", "📂 Upload File (.txt)"])
    
    input_data = None
    input_type = None
    input_title = None # Tên hiển thị trong lịch sử
    
    # 1. Xử lý Tab Link
    with tab_link:
        url = st.text_input("Dán đường dẫn bài báo:", placeholder="https://...")
        if st.button("Phân tích Link"):
            if url:
                with st.spinner("Đang đọc nội dung từ web..."):
                    title, content, err = crawl_news_from_url(url)
                    if err: st.error(err)
                    else:
                        input_data = content
                        input_type = "Link"
                        input_title = title

    # 2. Xử lý Tab Text
    with tab_text:
        txt = st.text_area("Dán nội dung vào đây:", height=100)
        if st.button("Phân tích Văn bản"):
            if txt:
                input_data = txt
                input_type = "Text"
                input_title = f"Văn bản ({txt[:30]}...)"

    # 3. Xử lý Tab File
    with tab_file:
        uploaded_file = st.file_uploader("Chọn file .txt", type="txt")
        if uploaded_file is not None and st.button("Phân tích File"):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = stringio.read()
            if content:
                input_data = content
                input_type = "File"
                input_title = uploaded_file.name

    # --- CORE: XỬ LÝ & LƯU VÀO SESSION STATE ---
    if input_data:
        # 1. Preprocess
        clean_text = preprocess_text(input_data)
        
        # 2. Predict
        vec = tfidf.transform([clean_text])
        probs = model.predict_proba(vec)[0]
        pred_idx = np.argmax(probs)
        label = le.inverse_transform([pred_idx])[0]
        conf = probs[pred_idx]
        
        # 3. Lưu vào lịch sử
        new_entry = {
            "title": input_title,
            "type": input_type,
            "topic": label,
            "confidence": conf,
            "preview": input_data[:100] + "..."
        }
        st.session_state['history'].append(new_entry)
        st.success(f"Đã thêm: **{label}** ({conf:.1%})")
        # Không rerun để người dùng có thể nhập tiếp liên tục

# --- KHU VỰC 2: DASHBOARD XU HƯỚNG ---
st.divider()

if len(st.session_state['history']) > 0:
    st.subheader("📈 Xu hướng đọc của bạn")
    
    # Chuyển lịch sử thành DataFrame để dễ xử lý
    df_history = pd.DataFrame(st.session_state['history'])
    
    # 1. KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số bài đã đọc", len(df_history))
    
    top_topic = df_history['topic'].mode()[0]
    col2.metric("Chủ đề quan tâm nhất", top_topic)
    
    avg_conf = df_history['confidence'].mean()
    col3.metric("Độ tin cậy trung bình AI", f"{avg_conf:.1%}")
    
    # 2. Charts & Details
    c_chart, c_list = st.columns([1, 1])
    
    with c_chart:
        st.write("##### Phân bố chủ đề")
        # Vẽ Pie Chart
        topic_counts = df_history['topic'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = sns.color_palette('pastel')[0:len(topic_counts)]
        ax.pie(topic_counts, labels=topic_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        st.pyplot(fig)

    with c_list:
        st.write("##### Lịch sử chi tiết")
        # Hiển thị dạng bảng rút gọn
        st.dataframe(
            df_history[['topic', 'title', 'type', 'confidence']].style.highlight_max(axis=0, subset=['confidence']),
            column_config={
                "topic": "Chủ đề",
                "title": "Nguồn / Tiêu đề",
                "type": "Loại",
                "confidence": st.column_config.NumberColumn("Độ tin cậy", format="%.2f")
            },
            use_container_width=True,
            height=300
        )

else:

    st.info("👈 Hãy nhập Link, Văn bản hoặc File ở trên để xem Dashboard phân tích xu hướng.")
