import os
import shutil

# Lấy đường dẫn hiện tại của file script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Thư mục chứa dữ liệu gốc (Input)
SRC_ROOT = os.path.join(BASE_DIR, "data")
# Thư mục chứa dữ liệu sau khi đổi tên (Output)
DST_ROOT = os.path.join(BASE_DIR, "data_filtered")

# ========= 1. MAPPING TOPIC ABBR (Tên Topic -> Mã viết tắt) =========
TOPIC_ABBR_FIXED = {
    "Thế giới": "TG",
    "Pháp luật": "PL",
    "Kinh doanh": "KD",
    "Bất động sản": "BĐS",
    "Chứng khoán": "CK",
    "Công nghệ": "CN",
    "Khoa học": "KH",
    "Sức khỏe": "SK",
    "Giáo dục": "GD",
    "Thể thao": "TT",
    "Giải trí": "GTr",
    "Du lịch": "DL",
    "Ẩm thực": "AT",
    "Gia đình": "GĐ",
    "Văn hóa": "VH",
    "Đời sống": "ĐS",
    "Nông nghiệp": "NN",
    "Thời sự - Chính trị": "TSCT",
    "Giao thông": "GT",
    "Khởi nghiệp": "KN",
    "Xã hội": "XH",
}

# ========= 2. MAPPING SOURCE ABBR (Tên Nguồn -> Mã viết tắt) =========
SOURCE_ABBR_MAP = {
    # Lao Động
    "ld": "LD", "laodong": "LD",
    # Người Lao Động
    "nld": "NLD", "nguoilaodong": "NLD",
    # Tuổi Trẻ
    "tt": "TT", "tuoitre": "TT",
    # Thanh Niên
    "tn": "TN", "thanhnien": "TN",
    # VnExpress
    "vne": "VNE", "vnexpress": "VNE",
    # Vietnamnet
    "vnn": "VNN", "vietnamnet": "VNN",
    # Zing
    "zing": "ZING", "znews": "ZING",
    # VOV, VTV
    "vov": "VOV", "vtv": "VTV",
    # Dân trí
    "dt": "DT", "dantri": "DT",
    # CafeF
    "cafef": "CAF", "cafe": "CAF",
    # Vietstock
    "vst": "VST", "vietstock": "VST",
    # Báo Nông nghiệp
    "baonn": "BNN", "bnn": "BNN", "baonongnghiep": "BNN",
    "nongnghiep": "BNN", "nongnghiepvn": "BNN",
}


# ========= 3. CÁC HÀM HỖ TRỢ =========

def get_topic_abbr(topic_folder: str) -> str:
    """Lấy mã viết tắt của Topic"""
    return TOPIC_ABBR_FIXED.get(topic_folder, "TP")


def normalize_source_token(token: str) -> str:
    """Chuẩn hóa tên nguồn (VD: dantri -> DT)"""
    raw = token.strip()
    compact = raw.replace(" ", "").replace("-", "")
    key = compact.lower()
    return SOURCE_ABBR_MAP.get(key, compact.upper())


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def process_topic(topic_folder: str):
    src_dir = os.path.join(SRC_ROOT, topic_folder)
    dst_dir = os.path.join(DST_ROOT, topic_folder)

    # Nếu thư mục nguồn không tồn tại thì bỏ qua
    if not os.path.isdir(src_dir):
        return

    # Tạo thư mục đích
    ensure_dir(dst_dir)

    # Lấy mã viết tắt Topic (VD: Bất động sản -> BĐS)
    topic_abbr = get_topic_abbr(topic_folder)

    # Lấy danh sách file .txt và sắp xếp để thứ tự ổn định
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".txt")]
    files.sort()

    if not files:
        print(f"[INFO] Topic '{topic_folder}' trống, bỏ qua.")
        return

    print(f"\n=== Đang xử lý Topic: '{topic_folder}' (Mã: {topic_abbr}) ===")
    print(f"  - Tìm thấy: {len(files)} file")

    # [QUAN TRỌNG] Khởi tạo biến đếm index = 1 cho TOÀN BỘ topic này.
    # Biến này sẽ tăng dần và KHÔNG reset khi gặp source khác.
    current_index = 1

    for fname in files:
        src_path = os.path.join(src_dir, fname)

        # Giả sử tên file cũ dạng: TOPIC_SOURCE_ID.txt hoặc SOURCE_...
        # Ta cố gắng tách lấy phần Source (thường là phần tử thứ 2 sau dấu gạch dưới)
        base, ext = os.path.splitext(fname)
        parts = [p for p in base.split("_") if p]

        # Logic lấy Source:
        # Nếu tên file có dấu gạch dưới (VD: DL_VNE_123.txt) -> Lấy 'VNE' (parts[1])
        # Nếu không có (VD: dantri.txt) -> Lấy 'dantri' (parts[0])
        if len(parts) >= 2:
            source_token = parts[1]
        else:
            source_token = parts[0] if parts else "UNK"

        # Chuẩn hóa Source (VD: vnexpress -> VNE)
        source_abbr = normalize_source_token(source_token)

        # Tạo tên file mới theo format: TOPIC_SOURCE_INDEX.txt
        # Ví dụ: BĐS_VNE_1.txt
        new_name = f"{topic_abbr}_{source_abbr}_{current_index}.txt"
        dst_path = os.path.join(dst_dir, new_name)

        # Đọc file nguồn và ghi sang file đích (để đảm bảo UTF-8 và lọc lỗi)
        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as f_in:
                content = f_in.read()

            with open(dst_path, "w", encoding="utf-8") as f_out:
                f_out.write(content)
        except Exception as e:
            print(f"  [LỖI] Không thể xử lý file {fname}: {e}")
            continue

        # Tăng biến đếm lên 1 cho file tiếp theo
        current_index += 1

    print(f"  -> Đã xong. File cuối cùng là số: {current_index - 1}")


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"[LỖI] Thư mục nguồn không tồn tại: {SRC_ROOT}")
        print("Vui lòng đảm bảo bạn đã tạo thư mục 'data' và bỏ các folder topic vào đó.")
        return

    # Xóa thư mục đích cũ nếu muốn sạch sẽ (tùy chọn), ở đây ta chỉ ensure nó tồn tại
    ensure_dir(DST_ROOT)

    # Lấy danh sách các thư mục Topic
    topic_folders = [
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    ]
    topic_folders.sort()

    print(f"Nguồn dữ liệu: {SRC_ROOT}")
    print(f"Nơi lưu kết quả: {DST_ROOT}")
    print(f"Số lượng Topic: {len(topic_folders)}")

    for topic in topic_folders:
        process_topic(topic)

    print("\n=== HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ===")


if __name__ == "__main__":
    main()