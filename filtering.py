import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(BASE_DIR, "data")
DST_ROOT = os.path.join(BASE_DIR, "data_filtered")

# ========= 1. MAPPING TOPIC ABBR =========
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

# ========= 2. MAPPING SOURCE ABBR =========
SOURCE_ABBR_MAP = {
    # Lao Động
    "ld": "LD",
    "laodong": "LD",

    # Người Lao Động
    "nld": "NLD",
    "nguoilaodong": "NLD",

    # Tuổi Trẻ
    "tt": "TT",
    "tuoitre": "TT",

    # Thanh Niên
    "tn": "TN",
    "thanhnien": "TN",

    # VnExpress
    "vne": "VNE",
    "vnexpress": "VNE",

    # Vietnamnet
    "vnn": "VNN",
    "vietnamnet": "VNN",

    # Zing
    "zing": "ZING",
    "znews": "ZING",

    # VOV
    "vov": "VOV",

    # VTV
    "vtv": "VTV",

    # Dân trí
    "dt": "DT",
    "dantri": "DT",

    # CafeF → CAF
    "cafef": "CAF",
    "cafe": "CAF",

    # Vietstock
    "vst": "VST",
    "vietstock": "VST",

    # Báo Nông nghiệp → BNN
    "baonn": "BNN",
    "bnn": "BNN",
    "baonongnghiep": "BNN",
    "nongnghiep": "BNN",
    "nongnghiepvn": "BNN",
}


# ========= 3. HÀM HỖ TRỢ =========
def auto_topic_abbr(topic_name: str) -> str:
    """
    Nếu topic không có trong TOPIC_ABBR_FIXED thì tự sinh:
    - Lấy các từ (split theo khoảng trắng và '-')
    - Lấy chữ cái đầu mỗi từ, ghép lại, upper.
    """
    cleaned = topic_name.strip()
    if not cleaned:
        return "TP"  # fallback an toàn, không để UNK

    parts = [p for p in cleaned.replace("-", " ").split() if p]
    initials = "".join(p[0] for p in parts)
    return initials.upper() if initials else "TP"


def get_topic_abbr(topic_folder: str) -> str:
    if topic_folder in TOPIC_ABBR_FIXED:
        return TOPIC_ABBR_FIXED[topic_folder]
    return auto_topic_abbr(topic_folder)


def normalize_source_token(token: str) -> str:
    raw = token.strip()
    compact = raw.replace(" ", "").replace("-", "")
    key = compact.lower()
    if key in SOURCE_ABBR_MAP:
        return SOURCE_ABBR_MAP[key]
    return compact.upper() if compact else "UNK"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_topic(topic_folder: str):
    src_dir = os.path.join(SRC_ROOT, topic_folder)
    dst_dir = os.path.join(DST_ROOT, topic_folder)

    if not os.path.isdir(src_dir):
        return

    ensure_dir(dst_dir)

    topic_abbr = get_topic_abbr(topic_folder)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(".txt")]
    files.sort()
    if not files:
        print(f"[INFO] Topic '{topic_folder}' không có file .txt.")
        return

    print(f"\n=== TOPIC: '{topic_folder}' -> abbr '{topic_abbr}' ===")
    print(f"  - Số file: {len(files)}")

    index = 1
    for fname in files:
        src_path = os.path.join(src_dir, fname)

        base, ext = os.path.splitext(fname)
        parts = [p for p in base.split("_") if p]

        if len(parts) >= 2:
            source_token = parts[1]
        else:
            source_token = base  # fallback

        source_abbr = normalize_source_token(source_token)

        new_name = f"{topic_abbr}_{source_abbr}_{index}.txt"
        dst_path = os.path.join(dst_dir, new_name)

        with open(src_path, "r", encoding="utf-8", errors="ignore") as f_in:
            content = f_in.read()
        with open(dst_path, "w", encoding="utf-8") as f_out:
            f_out.write(content)

        index += 1

    print(f"  - Đã ghi {index - 1} file sang '{dst_dir}'")


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"Folder nguồn không tồn tại: {SRC_ROOT}")
        return

    ensure_dir(DST_ROOT)

    topic_folders = [
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d))
    ]
    topic_folders.sort()

    print(f"SRC_ROOT: {SRC_ROOT}")
    print(f"DST_ROOT: {DST_ROOT}")
    print(f"Tìm thấy {len(topic_folders)} topic folder: {topic_folders}")

    for topic in topic_folders:
        process_topic(topic)


if __name__ == "__main__":
    main()
