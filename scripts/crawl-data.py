import os
import time
from pathlib import Path
from bs4 import BeautifulSoup
from newspaper import Article

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- CẤU HÌNH ---
DATA_ROOT = Path("Data_BatDongSan")
SCROLL_PAUSE_TIME = 2
MAX_SCROLL_ATTEMPTS = 5

SOURCES = [
    {
        "topic": "DuLich",
        "abbr": "DL",
        "name": "VNN",
        "url": "https://vietnamnet.vn/du-lich",
        "container": ".content-detail"
    },
    {
        "topic": "DuLich",
        "abbr": "DL",
        "name": "VOV",
        "url": "https://vov.vn/du-lich",
        "container": ".article-content"
    }
    # Bạn có thể thêm các nguồn khác vào đây
]

def setup_driver():
    """Khởi tạo Chrome Driver với các tùy chọn cơ bản."""
    options = Options()
    # options.add_argument("--headless") # Bỏ comment nếu muốn chạy ẩn
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")
    
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def scroll_to_bottom(driver):
    """Cuộn trang xuống cuối để tải thêm nội dung."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    attempts = 0
    
    while attempts < MAX_SCROLL_ATTEMPTS:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            attempts += 1
        else:
            last_height = new_height
            attempts = 0

def extract_content(driver, url, container_selector):
    """Mở tab mới, lấy nội dung bài viết và đóng tab."""
    original_window = driver.current_window_handle
    driver.switch_to.new_window('tab')
    driver.get(url)
    time.sleep(2) # Đợi trang tải

    content = ""
    try:
        # Cách 1: Dùng Newspaper3k
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
        
        # Cách 2: Nếu Newspaper lấy ít quá, dùng Selenium + BeautifulSoup
        if len(content) < 200:
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            element = soup.select_one(container_selector)
            if element:
                content = element.get_text(separator="\n", strip=True)
                
    except Exception as e:
        print(f"Error extracting {url}: {e}")
    finally:
        driver.close()
        driver.switch_to.window(original_window)
        
    return content

def save_to_file(folder_path, filename, url, content):
    """Lưu nội dung vào file txt."""
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / filename
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Url: {url}\n\n{content}")

def process_source(source_config):
    """Xử lý logic crawl cho một nguồn cụ thể."""
    driver = setup_driver()
    count = 0
    
    try:
        print(f"Starting crawl: {source_config['name']}")
        driver.get(source_config['url'])
        scroll_to_bottom(driver)
        
        # Lấy danh sách link bài viết
        elems = driver.find_elements(By.TAG_NAME, "a")
        links = {e.get_attribute('href') for e in elems if e.get_attribute('href')}
        
        # Lọc link hợp lệ (đơn giản hóa regex)
        # Tùy chỉnh logic lọc link ở đây nếu cần thiết
        valid_links = [l for l in links if len(l) > 20 and source_config['url'] in l]

        for link in valid_links:
            content = extract_content(driver, link, source_config['container'])
            
            if len(content) > 100:
                filename = f"{source_config['abbr']}_{source_config['name']}_{int(time.time())}_{count}.txt"
                save_folder = DATA_ROOT / source_config['topic']
                save_to_file(save_folder, filename, link, content)
                count += 1
                print(f"Saved: {filename}")

    except Exception as e:
        print(f"Error processing source {source_config['name']}: {e}")
    finally:
        driver.quit()
        print(f"Finished {source_config['name']}. Total saved: {count}")

if __name__ == "__main__":
    for source in SOURCES:
        process_source(source)