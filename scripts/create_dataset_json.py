import json
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted

# --- CẤU HÌNH ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "processed" / "data_filtered"
OUTPUT_FILE = BASE_DIR / "data" / "final" / "nlp_dataset.jsonl"
MAPPING_FILE = BASE_DIR / "data" / "final" / "id2label.json"

def create_label_mapping(data_dir):
    """Tạo dictionary mapping giữa tên label và ID số."""
    # Lấy danh sách thư mục con và sắp xếp tự nhiên
    topic_names = natsorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    
    label2id = {name: idx for idx, name in enumerate(topic_names)}
    id2label = {idx: name for idx, name in enumerate(topic_names)}
    
    return label2id, id2label

def process_dataset():
    if not RAW_DATA_DIR.exists():
        print(f"Directory not found: {RAW_DATA_DIR}")
        return

    # 1. Tạo và lưu file mapping
    label2id, id2label = create_label_mapping(RAW_DATA_DIR)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False, indent=4)
    
    print(f"Found {len(label2id)} topics. Mapping saved.")

    # 2. Đọc file txt và ghi vào jsonl
    files = sorted(RAW_DATA_DIR.rglob("*.txt"))
    
    print("Processing files...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for file_path in tqdm(files):
            try:
                # Tên thư mục cha chính là nhãn (label)
                label_name = file_path.parent.name
                
                # Bỏ qua nếu file nằm ngoài thư mục label hợp lệ
                if label_name not in label2id:
                    continue

                content = file_path.read_text(encoding='utf-8', errors='ignore').strip()
                
                if content:
                    record = {
                        "text": content,
                        "label_name": label_name,
                        "label_id": label2id[label_name],
                        "filename": file_path.name
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

    print(f"Dataset created at: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()