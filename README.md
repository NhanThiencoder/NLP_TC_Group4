# 🚀 Vietnamese Text Classification - NLP Group 4



![Python](https://img.shields.io/badge/Python-3.13.9%2B-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)

![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

![Status](https://img.shields.io/badge/Status-Active-success)



> Dự án Phân loại văn bản tiếng Việt.



## 👥 Thành viên nhóm (Group 4)



Dưới đây là danh sách các thành viên 

| 1 | **Trần Quốc Bảo** 

| 2 | **Nguyễn Ngọc Minh Đức**

| 3 | **Hồ Đức Nhân Thiện**

| 4 | **Nguyễn Huỳnh Tấn Phát**



---

## 📂 Cấu trúc dự án

```plaintext

NLP\_TC\_Group4/

├── data/                           # Chứa dữ liệu (Raw \& Processed)

├── models/                         # Chứa các checkpoint mô hình (được ignore trên git)

├── notebooks/                      # Jupyter Notebooks cho thực nghiệm

│   ├── EDA.ipynb                   # Phân tích khám phá dữ liệu

│   ├── preprocessing_data.ipynb    # Các bước tiền xử lý văn bản

│   └── model.ipynb                 # Huấn luyện và đánh giá mô hình

├── scripts/                        # Mã nguồn Python (Scripts)

│   ├── crawl-data.py               # Script thu thập dữ liệu

│   └── create_dataset_json.py      # Tạo dataset dưới kiểu json cho model

├── .gitignore                      # Cấu hình file ẩn

└── README.md                       # Tài liệu dự án
```

## 🎯 Mục tiêu dự án (Project Goal)

Mục tiêu chính của dự án là xây dựng một hệ thống **Phân loại văn bản tiếng Việt (Vietnamese Text Classification)** tự động và hiệu quả. Hệ thống được thiết kế để xử lý lượng lớn dữ liệu tin tức từ các báo điện tử, giúp tự động gán nhãn chủ đề mà không cần sự can thiệp thủ công của con người.

Dự án tập trung vào việc:
1.  **Xây dựng bộ dữ liệu chuẩn:** Thu thập và làm sạch dữ liệu văn bản tiếng Việt đa lĩnh vực.
2.  **Xử lý ngôn ngữ tự nhiên (NLP):** Áp dụng các kỹ thuật tiền xử lý đặc thù cho tiếng Việt (tách từ, chuẩn hóa).
3.  **Tối ưu hóa bài toán phân loại đa lớp:** Giải quyết thách thức phân loại văn bản với số lượng nhãn lớn (20 chủ đề) và độ tương đồng cao giữa các chủ đề.

## 💾 Danh sách chủ đề (Topics)

Tập dữ liệu bao gồm **20 nhóm chủ đề** đa dạng, đại diện cho các lĩnh vực quan trọng trong đời sống xã hội.

**Bảng phân bố 20 nhãn (Labels):**
| 📂 Ẩm thực | 📂 Gia đình | 📂 Khởi nghiệp | 📂 Thế giới |
| 📂 Bất động sản | 📂 Giao thông | 📂 Kinh doanh | 📂 Thể thao |
| 📂 Chứng khoán | 📂 Giáo dục | 📂 Nông nghiệp | 📂 Thời sự - Chính trị |
| 📂 Công nghệ | 📂 Giải trí | 📂 Pháp luật | 📂 Văn hóa |
| 📂 Du lịch | 📂 Khoa học | 📂 Sức khỏe | 📂 Đời sống |

## 🧠 Phương pháp tiếp cận (Methodology)

Quy trình thực hiện dự án được chia thành các giai đoạn chính:

1.  **Thu thập dữ liệu (Data Collection):** Sử dụng các script tự động để crawl dữ liệu từ các nguồn tin tức chính thống.
2.  **Tiền xử lý (Preprocessing):**
    * Làm sạch văn bản (loại bỏ HTML, ký tự rác).
    * Chuẩn hóa tiếng Việt (xử lý dấu câu, viết hoa/thường).
    * Tách từ (Word Segmentation) để giữ nguyên ngữ nghĩa của từ ghép.
3.  **Huấn luyện & Đánh giá (Training & Evaluation):**
    * Thử nghiệm các kỹ thuật Học sâu (Deep Learning) hiện đại để tìm ra phương án tối ưu nhất cho bài toán.
    * Đánh giá mô hình dựa trên các chỉ số: Accuracy, Precision, Recall và F1-Score trên tập kiểm thử độc lập.

---








