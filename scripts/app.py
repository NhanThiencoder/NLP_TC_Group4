import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import unicodedata
import requests
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ thống Phân loại Tin tức (TextMLP)",
    page_icon=None,
    layout="wide"
)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
STOPWORD_PATH = PROJECT_ROOT / "data" / "final" / "vietnamese-stopwords-dash.txt"

if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 1. ĐỊNH NGHĨA MODEL ---
class TextMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TextMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.out = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.out(x)

# --- 2. HÀM TẢI STOPWORDS ---
def load_stopwords(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f.readlines()])
    except FileNotFoundError:
        return {"thì", "là", "mà", "và", "của", "những"}

STOPWORDS = load_stopwords(STOPWORD_PATH)

# --- 3. TIỀN XỬ LÝ ---
def normalize_text(text):
    return unicodedata.normalize('NFC', text)

def preprocess_text(text):
    text = normalize_text(text)
    tokenized = ViTokenizer.tokenize(text)
    words = tokenized.split()
    clean_words = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(clean_words)

def crawl_news_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else "Bài viết từ Link"
        paragraphs = soup.find_all('p', class_=['Normal', 'description', 'content', 'fck_detail', 'article-content'])
        if not paragraphs: paragraphs = soup.find_all('p') 
        
        content = "\n".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        
        if len(content) < 100: return None, None, "Nội dung quá ngắn hoặc không crawl được."
        return title, content, None
    except Exception as e: return None, None, str(e)

# --- 4. LOAD MODEL & RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        tfidf_path = MODEL_DIR / "tfidf_vectorizer.pkl"
        le_path = MODEL_DIR / "label_encoder.pkl"
        model_path = MODEL_DIR / "mlp_model.pth"

        if not (tfidf_path.exists() and le_path.exists() and model_path.exists()):
            st.error(f"Thiếu file trong thư mục: {MODEL_DIR}")
            return None, None, None

        le = joblib.load(le_path)
        tfidf = joblib.load(tfidf_path)
        
        input_dim = len(tfidf.get_feature_names_out())
        num_classes = len(le.classes_)
        
        model = TextMLP(input_size=input_dim, num_classes=num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return le, tfidf, model
    except Exception as e:
        st.error(f"Lỗi load model: {str(e)}")
        return None, None, None

le, tfidf, model = load_resources()

# --- GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.title("Cài đặt")
    if st.button("Xóa lịch sử", type="primary"):
        st.session_state['history'] = []
        st.rerun()
    st.info("Mô hình: TextMLP (3 Layers + BN)")

st.title("Phân loại Tin tức (MLP)")

if not model:
    st.stop()

with st.container(border=True):
    st.subheader("Nhập dữ liệu")
    tab_link, tab_text, tab_file = st.tabs(["Link Báo", "Văn bản", "File .txt"])
    
    input_data, input_type, input_title = None, None, None
    
    with tab_link:
        url = st.text_input("Dán URL bài báo:")
        if st.button("Phân tích Link"):
            if url:
                with st.spinner("Đang crawl dữ liệu..."):
                    title, content, err = crawl_news_from_url(url)
                    if err: st.error(err)
                    else:
                        input_data = content
                        input_type = "Link"
                        input_title = title

    with tab_text:
        txt = st.text_area("Dán nội dung văn bản:")
        if st.button("Phân tích Text"):
            if txt:
                input_data = txt
                input_type = "Text"
                input_title = f"Text ({txt[:20]}...)"

    with tab_file:
        uploaded_file = st.file_uploader("Chọn file .txt", type="txt")
        if uploaded_file and st.button("Phân tích File"):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            input_data = stringio.read()
            input_type = "File"
            input_title = uploaded_file.name

    if input_data:
        clean_text = preprocess_text(input_data)
        
        vec = tfidf.transform([clean_text]).toarray()
        tensor_input = torch.tensor(vec, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
            label = le.inverse_transform([pred_idx.item()])[0]
            confidence = conf.item()
        
        st.session_state['history'].append({
            "title": input_title,
            "type": input_type,
            "topic": label,
            "confidence": confidence
        })
        st.success(f"Dự đoán: **{label}**")
        st.progress(confidence, text=f"Độ tin cậy: {confidence:.1%}")

st.divider()

# --- DASHBOARD (ĐÃ CHỈNH SỬA) ---
if len(st.session_state['history']) > 0:
    st.subheader("Thống kê phiên làm việc")
    df = pd.DataFrame(st.session_state['history'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Tổng số bài", len(df))
    c2.metric("Chủ đề chính", df['topic'].mode()[0])
    c3.metric("Độ tin cậy TB", f"{df['confidence'].mean():.1%}")
    
    # --- THAY ĐỔI Ở ĐÂY: Tỷ lệ 1:1 (Chart bự hơn, Bảng nhỏ lại) ---
    col_chart, col_list = st.columns([1, 1]) 
    
    with col_chart:
        counts = df['topic'].value_counts()
        # Tăng kích thước hình (figsize) lên 6x6
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)
        
    with col_list:
        st.dataframe(
            df[['topic', 'title', 'confidence']].style.format({"confidence": "{:.1%}"}), 
            use_container_width=True,
            height=300 # Tăng chiều cao bảng cho cân đối với biểu đồ
        )