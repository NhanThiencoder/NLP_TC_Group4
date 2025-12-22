import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import unicodedata
import requests
import io
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Personal Content Analyzer (TextCNN)",
    page_icon="🧠",
    layout="wide"
)

# --- CẤU HÌNH ĐƯỜNG DẪN (Chạy trong folder scripts) ---
CURRENT_DIR = Path(__file__).parent 
BASE_DIR = CURRENT_DIR.parent 
MODEL_DIR = BASE_DIR / "models"
if not MODEL_DIR.exists(): MODEL_DIR = CURRENT_DIR / "models" # Fallback

# --- 1. KHỞI TẠO SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- 2. ĐỊNH NGHĨA MODEL TEXT-CNN (MODEL CHÍNH) ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[2, 3, 4], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Tạo 3 lớp Conv song song quét các cửa sổ 2 từ, 3 từ, 4 từ
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        x = self.embedding(x)             # [Batch, Seq_Len, Embed]
        x = x.unsqueeze(1)                # [Batch, 1, Seq_Len, Embed] -> Thêm channel dimension
        
        # Qua Conv + ReLU + MaxPool
        # Kết quả là danh sách các tensor đã được pool
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        
        # Nối lại và qua lớp Fully Connected
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# --- 3. CÁC HÀM XỬ LÝ DỮ LIỆU ---
STOPWORDS = {
    "thì", "là", "mà", "của", "những", "các", "để", "và", "với", "có", 
    "trong", "đã", "đang", "sẽ", "được", "bị", "tại", "vì", "như", "này",
    "cho", "về", "một", "người", "khi", "ra", "vào", "lên", "xuống",
    "tôi", "chúng_tôi", "bạn", "họ", "chúng_ta", "theo", "ông", "bà",
    "nhiều", "ít", "rất", "quá", "lắm", "nhưng", "tuy_nhiên", "nếu", "dù",
    "bài", "viết", "ảnh", "video", "clip", "nguồn", "theo", "vnexpress", "dân trí"
}

def normalize_text(text): return unicodedata.normalize('NFC', text)

def preprocess_text(text):
    text = normalize_text(text)
    tokenized = ViTokenizer.tokenize(text)
    words = tokenized.split()
    clean_words = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    return " ".join(clean_words)

def text_to_sequence(text, vocab, max_len=1024):
    # Chuyển text thành chuỗi số ID dựa trên vocab
    seq = [vocab.get(w, 1) for w in text.split()] # 1 is <UNK>
    # Padding hoặc Cắt
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq)) # 0 is <PAD>
    else:
        seq = seq[:max_len]
    return seq

def crawl_news_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else "Link không tiêu đề"
        
        # Lấy nội dung thông minh
        paragraphs = soup.find_all('p', class_=['Normal', 'description', 'content', 'detail-content'])
        if not paragraphs: paragraphs = soup.find_all('p') 
        
        content = "\n".join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])
        
        if len(content) < 100: return None, None, "Nội dung quá ngắn (có thể bị chặn hoặc web dùng JS)."
        return title, content, None
    except Exception as e: return None, None, str(e)

# --- 4. LOAD MODELS (CACHE) ---
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # 1. Load Label Encoder
        le = joblib.load(MODEL_DIR / "label_encoder.pkl")
        
        # 2. Load TextCNN Model
        # Lưu ý: Tên file phải khớp với lúc bạn save trong model.ipynb (ví dụ: textcnn_model.pth)
        # Nếu bạn save tên khác, hãy sửa lại dòng này
        checkpoint_path = MODEL_DIR / "textcnn_model.pth" 
        
        if not checkpoint_path.exists():
            st.error(f"Không tìm thấy file: {checkpoint_path}")
            return None, None, None, None
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vocab = checkpoint['vocab']
        config = checkpoint['config']
        
        model = TextCNN(
            vocab_size=config['vocab_size'], 
            embed_dim=config['embed_dim'], 
            num_classes=config['num_classes'],
            filter_sizes=config.get('filter_sizes', [2,3,4]),
            num_filters=config.get('num_filters', 100)
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        
        return le, model, vocab, config
        
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None, None, None, None

le, cnn_model, vocab, cnn_config = load_resources()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 5. GIAO DIỆN CHÍNH ---
with st.sidebar:
    st.title("⚙️ Điều khiển")
    if st.button("🗑️ Xóa dữ liệu", type="primary"):
        st.session_state['history'] = []
        st.rerun()
    st.info(f"Model: **TextCNN**\nDevice: {device}")
    st.caption("TextCNN vượt trội nhờ khả năng bắt các cụm từ cục bộ (n-grams) quan trọng.")

st.title("🚀 Smart Content Analytics")
st.markdown("Hệ thống phân tích xu hướng đọc sử dụng **TextCNN Deep Learning**.")

if not cnn_model:
    st.warning("⚠️ Đang chạy ở chế độ Demo giao diện (Chưa load được Model).")
    st.stop()

# --- INPUT AREA ---
with st.container(border=True):
    st.subheader("📥 Nhập nội dung phân tích")
    
    tab1, tab2, tab3 = st.tabs(["🔗 Link Báo", "📝 Văn bản", "📂 File Text"])
    
    input_payload = None
    input_source = ""
    
    with tab1:
        url = st.text_input("URL bài báo:", placeholder="https://vnexpress.net/...")
        if st.button("Phân tích Link"):
            if url:
                with st.spinner("Đang cào dữ liệu..."):
                    t, c, e = crawl_news_from_url(url)
                    if e: st.error(e)
                    else:
                        input_payload = c
                        input_source = t
    
    with tab2:
        txt = st.text_area("Nội dung:", height=100)
        if st.button("Phân tích Text"):
            if txt:
                input_payload = txt
                input_source = f"Văn bản ({txt[:20]}...)"
                
    with tab3:
        f = st.file_uploader("Chọn file .txt", type="txt")
        if f and st.button("Phân tích File"):
            stringio = io.StringIO(f.getvalue().decode("utf-8"))
            input_payload = stringio.read()
            input_source = f.name

    # --- CORE PREDICTION LOGIC ---
    if input_payload:
        # 1. Preprocess
        clean_text = preprocess_text(input_payload)
        
        # 2. Vectorize (Sequence)
        max_len = cnn_config.get('max_len', 1024)
        seq = text_to_sequence(clean_text, vocab, max_len)
        tensor_in = torch.tensor([seq], dtype=torch.long).to(device)
        
        # 3. Predict with TextCNN
        with torch.no_grad():
            logits = cnn_model(tensor_in)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
            
            label = le.inverse_transform([idx.item()])[0]
            confidence = conf.item()
            
        # 4. Save to History
        st.session_state['history'].append({
            "source": input_source,
            "topic": label,
            "conf": confidence,
            "timestamp": pd.Timestamp.now()
        })
        
        st.success(f"Kết quả: **{label}** ({confidence:.1%})")

# --- DASHBOARD AREA ---
st.divider()

if st.session_state['history']:
    st.subheader("📊 Dashboard Xu hướng của bạn")
    
    df = pd.DataFrame(st.session_state['history'])
    
    # KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Tổng bài đã đọc", len(df))
    k2.metric("Chủ đề Top 1", df['topic'].mode()[0])
    k3.metric("Độ tin cậy AI", f"{df['conf'].mean():.1%}")
    
    # Charts
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.caption("Phân bố chủ đề")
        counts = df['topic'].value_counts()
        fig, ax = plt.subplots(figsize=(5,5))
        colors = sns.color_palette('pastel')[0:len(counts)]
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
        st.pyplot(fig)
        
    with c2:
        st.caption("Lịch sử chi tiết")
        st.dataframe(
            df[['topic', 'source', 'conf']].style.highlight_max(subset=['conf'], color='#d1e7dd'),
            column_config={
                "topic": "Chủ đề",
                "source": "Nguồn",
                "conf": st.column_config.NumberColumn("Độ tin cậy", format="%.2f")
            },
            use_container_width=True,
            height=300
        )
else:
    st.info("Dữ liệu phân tích sẽ xuất hiện ở đây sau khi bạn nhập bài viết.")