import json
import os
from pathlib import Path

from natsort import natsorted
from tqdm import tqdm
import natsort

# ================= CẤU HÌNH =================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "data_filtered"
OUTPUT_FILE = BASE_DIR / "data" / "final" / "nlp_dataset.jsonl"
MAPPING_FILE = BASE_DIR / "data" / "final" / "id2label.json"


def create_dataset_jsonl():
    if not DATA_DIR.exists():
        print(f"❌ Không tìm thấy thư mục: {DATA_DIR}")
        return

    # 1. Tạo Mapping ID
    topics = natsorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    label2id = {name: idx for idx, name in enumerate(topics)}
    id2label = {idx: name for idx, name in enumerate(topics)}

    print(f"📊 Tìm thấy {len(topics)} chủ đề.")

    # 2. Lưu file Mapping (Để sau này biết số 0 là topic gì)
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu file mapping: {MAPPING_FILE.name}")

    # 3. Duyệt file và Ghi trực tiếp vào JSONL (Stream write)
    print(f"🚀 Đang tạo dataset {OUTPUT_FILE.name}...")

    total_files = sum(len(list(d.glob("*.txt"))) for d in DATA_DIR.iterdir() if d.is_dir())

    # Mở file dataset để ghi dòng (Append mode)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        with tqdm(total=total_files, unit="file") as pbar:
            for topic_name in topics:
                topic_dir = DATA_DIR / topic_name
                topic_id = label2id[topic_name]

                # Lấy file và sort
                files = sorted(topic_dir.glob("*.txt"))

                for file_path in files:
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore').strip()
                        if content:
                            # Tạo object
                            record = {
                                "text": content,
                                "label_name": topic_name,
                                "label_id": topic_id,
                                "filename": file_path.name  # Lưu thêm tên file gốc để dễ trace
                            }
                            # Ghi ngay lập tức 1 dòng JSON vào file
                            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')

                    except Exception as e:
                        print(f"[ERR] {file_path.name}: {e}")

                    pbar.update(1)

    print(f"\n✅ HOÀN TẤT! File dataset đã sẵn sàng để train.")


if __name__ == "__main__":
    create_dataset_jsonl()