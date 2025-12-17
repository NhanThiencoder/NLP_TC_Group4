import shutil
import os
from pathlib import Path
from huggingface_hub import HfApi, login

# --- CẤU HÌNH ---
REPO_ID = "JackStar2/NLP-20Topics-Articles"

# --- TỰ ĐỘNG XÁC ĐỊNH ĐƯỜNG DẪN ---
# Vì file này nằm trong folder 'scripts', ta cần lùi ra 1 cấp để thấy folder 'data'
BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Đường dẫn đến folder chứa 127k file txt (Data Processed)
FOLDER_DATA_SOURCE = BASE_DIR / "data" / "processed" / "data_filtered"

# 2. Đường dẫn đến file JSONL và Mapping (Data Final)
FILE_JSONL = BASE_DIR / "data" / "final" / "nlp_dataset.jsonl"
FILE_MAPPING = BASE_DIR / "data" / "final" / "id2label.json"

# Tên file nén tạm thời
ZIP_OUTPUT_NAME = "data_filtered_backup"

def main():
    # 1. Đăng nhập
    print(f"🔑 Đăng nhập vào Hugging Face...")
    login(token=TOKEN)
    api = HfApi()

    # Tạo Repo nếu chưa có
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # ---------------------------------------------------------
    # PHẦN A: NÉN VÀ UPLOAD FOLDER DATA_FILTERED (BACKUP)
    # ---------------------------------------------------------
    if FOLDER_DATA_SOURCE.exists():
        print(f"\n📦 Đang nén folder '{FOLDER_DATA_SOURCE.name}'... (Có thể lâu)")

        # Tạo file zip tại thư mục gốc của project để dễ dọn dẹp
        shutil.make_archive(str(BASE_DIR / ZIP_OUTPUT_NAME), 'zip', FOLDER_DATA_SOURCE)
        zip_file_full = BASE_DIR / (ZIP_OUTPUT_NAME + ".zip")

        print(f"✅ Nén xong: {zip_file_full.name}")
        print(f"🚀 Đang upload ZIP lên Hugging Face...")

        try:
            api.upload_file(
                path_or_fileobj=zip_file_full,
                path_in_repo="raw_files/data_filtered.zip", # Để vào thư mục raw_files cho gọn
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            print("✅ Upload ZIP thành công!")

            # Xóa file zip tạm để giải phóng ổ cứng
            os.remove(zip_file_full)
            print("🧹 Đã dọn dẹp file zip tạm.")

        except Exception as e:
            print(f"❌ Lỗi upload ZIP: {e}")
    else:
        print(f"⚠️ Không tìm thấy folder: {FOLDER_DATA_SOURCE}")

    # ---------------------------------------------------------
    # PHẦN B: UPLOAD FILE JSONL (QUAN TRỌNG ĐỂ TRAIN)
    # ---------------------------------------------------------
    print("\n---------------------------------------------------------")
    print(f"🚀 Đang upload file dataset chuẩn (JSONL)...")

    files_to_upload = [FILE_JSONL, FILE_MAPPING]

    for file_path in files_to_upload:
        if file_path.exists():
            try:
                print(f"   -> Uploading {file_path.name}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path.name, # Để ngay root của repo
                    repo_id=REPO_ID,
                    repo_type="dataset"
                )
                print(f"   ✅ Xong {file_path.name}")
            except Exception as e:
                print(f"   ❌ Lỗi upload {file_path.name}: {e}")
        else:
            print(f"   ⚠️ Không tìm thấy {file_path.name} (Bỏ qua)")

    print("\n🎉 HOÀN TẤT TOÀN BỘ! Kiểm tra tại link:")
    print(f"👉 https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()