import os
import time
import re
from bs4 import BeautifulSoup
from newspaper import Article

# --- SELENIUM ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# =========================================================
# 1. CẤU HÌNH & DANH SÁCH NGUỒN
# =========================================================
DATA_ROOT_BASE = "Data_BatDongSan"

SOURCES = [
# ================= DU LỊCH (DL) =================
    #{ "topic": "DuLich", "abbr": "DL", "name": "DT", "url": "https://dantri.com.vn/du-lich.htm", "type": "page", "regex": r'\.htm$', "container": ".singular-content, .e-magazine__body" },
    { "topic": "DuLich", "abbr": "DL", "name": "VNN", "url": "https://vietnamnet.vn/du-lich", "type": "page", "regex": r'vietnamnet\.vn\/.*\.html$', "container": ".content-detail" },
    { "topic": "DuLich", "abbr": "DL", "name": "VOV", "url": "https://vov.vn/du-lich", "type": "page", "regex": r'vov\.vn\/.*\.vov$', "container": ".article-content" },

    # ================= GIA ĐÌNH (GD) =================
    { "topic": "GiaDinh", "abbr": "GD", "name": "TN", "url": "https://thanhnien.vn/doi-song/gia-dinh.htm", "type": "click", "regex": r'thanhnien\.vn\/.*\.htm$', "container": ".detail-content" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "TT", "url": "https://tuoitre.vn/gia-dinh.htm", "type": "click", "regex": r'tuoitre\.vn\/.*\.htm$', "container": ".detail-content" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "ZING", "url": "https://znews.vn/doi-song/gia-dinh.html", "type": "click", "regex": r'znews\.vn\/.*\.html$', "container": ".the-article-body" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "VNE", "url": "https://vnexpress.net/doi-song/to-am", "type": "page", "regex": r'-p\d+|to-am', "container": ".fck_detail" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "DT", "url": "https://dantri.com.vn/doi-song/gia-dinh.htm", "type": "page", "regex": r'\.htm$', "container": ".singular-content" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "VNN", "url": "https://vietnamnet.vn/doi-song/gia-dinh", "type": "page", "regex": r'vietnamnet\.vn\/.*\.html$', "container": ".content-detail" },
    { "topic": "GiaDinh", "abbr": "GD", "name": "VOV", "url": "https://vov.vn/doi-song/gia-dinh", "type": "page", "regex": r'vov\.vn\/.*\.vov$', "container": ".article-content" },
    # 1. VNEXPRESS (Page Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "VNE",  # Đã sửa abbr thành BĐS
        "url": "https://vnexpress.net/bat-dong-san",
        "type": "page",
        "regex": r'-p\d+|bat-dong-san',
        "container": ".fck_detail"
    },
    # 2. DÂN TRÍ (Page Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "DT",
        "url": "https://dantri.com.vn/bat-dong-san.htm",
        "type": "page",
        "regex": r'\.htm$',
        "container": ".singular-content, .e-magazine__body"
    },
    # 3. TUỔI TRẺ (Click/Scroll Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "TT",
        "url": "https://tuoitre.vn/bat-dong-san.htm",
        "type": "click",
        "regex": r'tuoitre\.vn\/.*\.htm$',
        "container": ".detail-content"
    },
    # 4. THANH NIÊN (Click/Scroll Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "TN",
        "url": "https://thanhnien.vn/bat-dong-san.htm",
        "type": "click",
        "regex": r'thanhnien\.vn\/.*\.htm$',
        "container": ".detail-content"
    },
    # 5. VIETNAMNET (Page Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "VNN",
        "url": "https://vietnamnet.vn/bat-dong-san",
        "type": "page",
        "regex": r'vietnamnet\.vn\/.*\.html$',
        "container": ".content-detail"
    },
    # 6. VOV (Page Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "VOV",
        "url": "https://vov.vn/kinh-te/dia-oc",
        "type": "page",
        "regex": r'vov\.vn\/.*\.vov$',
        "container": ".article-content"
    },
    # 7. ZINGNEWS (Click/Scroll Mode)
    {
        "topic": "BatDongSan", "abbr": "BĐS", "name": "ZING",
        "url": "https://znews.vn/bat-dong-san.html",
        "type": "click",
        "regex": r'znews\.vn\/.*\.html$',
        "container": ".the-article-body"
    }
]

TARGET_PER_SOURCE = 1000
MAX_PAGES = 100
MAX_CLICKS = 100


# =========================================================
# 2. HÀM QUẢN LÝ FILE
# =========================================================
def get_next_index(folder, topic_abbr, source_name):
    if not os.path.exists(folder): return 1
    max_idx = 0
    prefix = f"{topic_abbr}_{source_name}_"
    for f in os.listdir(folder):
        if f.startswith(prefix) and f.endswith(".txt"):
            try:
                match = re.search(r'_(\d+)\.txt$', f)
                if match:
                    num = int(match.group(1))
                    if num > max_idx: max_idx = num
            except:
                continue
    return max_idx + 1


# =========================================================
# 3. SETUP DRIVER
# =========================================================
def setup_driver():
    options = Options()
    options.add_argument("--disable-notifications")
    options.add_argument("--start-maximized")

    # Chiến thuật: none (Siêu nhanh, không đợi gì cả) hoặc eager
    # Với Dân Trí bị lỗi renderer, ta dùng 'eager' nhưng kết hợp try-catch
    options.page_load_strategy = 'eager'

    # --- CÁC CỜ CHỐNG LỖI RENDERER & TIMEOUT ---
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")  # Khắc phục lỗi thiếu bộ nhớ share
    options.add_argument("--no-sandbox")
    options.add_argument("--dns-prefetch-disable")  # Tắt tìm nạp DNS trước
    options.add_argument("--disable-features=NetworkService")  # Giúp ổn định hơn

    # Chặn ảnh triệt để
    prefs = {
        "profile.managed_default_content_settings.images": 2,
        "profile.default_content_setting_values.notifications": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        # Chặn cả CSS nếu cần (nhưng có thể làm hỏng layout lấy tin)
        "profile.managed_default_content_settings.cookies": 2,
        "profile.managed_default_content_settings.javascript": 1,  # Vẫn phải bật JS
        "profile.managed_default_content_settings.plugins": 2,
        "profile.managed_default_content_settings.popups": 2,
        "profile.managed_default_content_settings.geolocation": 2,
        "profile.managed_default_content_settings.media_stream": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # Tăng thời gian timeout lên 60s (để tránh lỗi -0.00x quá sớm)
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)

    return driver


# =========================================================
# 4. HÀM CRAWL & XỬ LÝ (ĐÃ SỬA ĐỂ QUẢN LÝ TAB)
# =========================================================

def extract_content_selenium(driver, url, container_selector):
    """
    Hàm này chỉ chạy khi đã ở trong Tab mới.
    """
    try:
        driver.get(url)
        time.sleep(1)  # Chờ load nhẹ
        text = driver.execute_script(f"""
            var container = document.querySelector('{container_selector}');
            return container ? container.innerText : '';
        """)
        return text
    except:
        return ""


def remove_ads(driver):
    try:
        driver.execute_script("""
            var selectors = ['iframe', '.ads', '.banner', '#sticky', '.sticky', '.video-box', 'header', '.cms-pagging'];
            selectors.forEach(s => {
                var els = document.querySelectorAll(s);
                els.forEach(e => e.remove());
            });
        """)
    except:
        pass


from selenium.common.exceptions import TimeoutException, WebDriverException


def get_links_page_mode(driver, source_cfg):
    collected_links = set()
    base_url = source_cfg['url']
    print(f"   📄 PAGE MODE: {source_cfg['name']}")

    # Biến đếm số lần liên tiếp không thấy link (để thoát nếu hết bài thật)
    empty_streak = 0

    for page in range(1, MAX_PAGES + 1):
        if len(collected_links) >= TARGET_PER_SOURCE + 10: break

        # Logic tạo URL
        if page == 1:
            current_url = base_url
        else:
            if "vnexpress" in base_url:
                current_url = f"{base_url}-p{page}"
            elif "dantri" in base_url:
                clean = base_url.replace(".htm", "").replace(".html", "")
                current_url = f"{clean}/trang-{page}.htm"
            else:
                sep = "&" if "?" in base_url else "?"
                current_url = f"{base_url}{sep}page={page}"

        # --- XỬ LÝ KẾT NỐI & CHECK LỖI 503 ---
        try:
            driver.get(current_url)
            time.sleep(2)  # Chờ load

            # 1. KIỂM TRA LỖI 503 / 403 / BẢO TRÌ
            page_title = driver.title.lower()
            page_src = driver.page_source.lower()

            if "503" in page_title or "service unavailable" in page_src or "server maintaining" in page_src:
                print(f"      🛑 PHÁT HIỆN LỖI 503 (Bị chặn/Server bận) tại trang {page}")
                print("      💤 Đang tạm nghỉ 60 giây để server mở lại...")
                time.sleep(60)  # Nghỉ 1 phút để "nguội" máy

                # Thử reload lại trang này một lần nữa
                driver.refresh()
                time.sleep(5)

                # Kiểm tra lại sau khi reload
                if "503" in driver.title:
                    print("      ❌ Vẫn bị chặn. Dừng nguồn này để tránh ban IP.")
                    break

        except TimeoutException:
            print(f"      ⚠️ Timeout trang {page} -> Ép dừng tải và quét tiếp.")
            try:
                driver.execute_script("window.stop();")
            except:
                pass
        except Exception as e:
            print(f"      ❌ Lỗi lạ: {e}")
            continue

        # --- QUÉT LINK ---
        try:
            html = driver.page_source
            raw_links = re.findall(r'href=["\'](.*?)["\']', html)

            count_new_in_page = 0
            for href in raw_links:
                if href.startswith("/"): href = "https://" + base_url.split("/")[2] + href
                if not href.startswith("http"): continue

                if re.search(source_cfg['regex'], href):
                    if not any(b in href for b in ['/video', '/podcast', '/media']):
                        if href not in collected_links:
                            collected_links.add(href)
                            count_new_in_page += 1

            if count_new_in_page == 0:
                print(f"      -> Không thấy link mới ở trang {page}.")
                empty_streak += 1
            else:
                empty_streak = 0  # Reset nếu tìm thấy bài

            # Nếu 3 trang liên tiếp không có bài nào -> Chắc chắn là hết bài hoặc lỗi -> Dừng
            if empty_streak >= 3:
                print("      🛑 Dừng quét vì 3 trang liên tiếp không có bài mới.")
                break

        except Exception as e:
            print(f"      ⚠️ Lỗi quét link: {e}")

    return list(collected_links)


def get_links_click_mode(driver, source_cfg):
    # (Giữ nguyên logic click/scroll của bạn)
    collected_links = set()
    url = source_cfg['url']
    print(f"   🖱️ CLICK/SCROLL MODE: {source_cfg['name']}")

    try:
        driver.get(url)
    except:
        pass
    time.sleep(3)

    BUTTON_XPATHS = [
        "//a[contains(text(), 'Xem thêm')]", "//button[contains(text(), 'Xem thêm')]",
        "//div[contains(@class, 'view-more')]//a", "//div[@class='list__viewmore']//a"
    ]

    for i in range(MAX_CLICKS):
        remove_ads(driver)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

        clicked = False
        for xpath in BUTTON_XPATHS:
            try:
                btns = driver.find_elements(By.XPATH, xpath)
                for btn in btns:
                    if btn.is_displayed():
                        driver.execute_script("arguments[0].click();", btn)
                        clicked = True
                        time.sleep(2)
                        break
                if clicked: break
            except:
                continue

        try:
            html = driver.page_source
            raw_links = re.findall(r'href=["\'](.*?)["\']', html)
            for href in raw_links:
                href = href.strip()
                if href.startswith("/"): href = "https://" + url.split("/")[2] + href
                if not href.startswith("http"): continue

                if re.search(source_cfg['regex'], href):
                    if not any(b in href for b in ['/video', '/podcast', '/media', 'javascript:']):
                        collected_links.add(href)
        except:
            pass
        if len(collected_links) >= TARGET_PER_SOURCE + 10: break
    return list(collected_links)


# =========================================================
# 5. CHƯƠNG TRÌNH CHÍNH (LOGIC TỐI ƯU RAM)
# =========================================================
if __name__ == "__main__":

    # ❌ KHÔNG setup driver ở ngoài vòng lặp
    # driver = setup_driver()

    for source in SOURCES:
        # ✅ SETUP DRIVER MỚI CHO TỪNG NGUỒN (XẢ RAM)
        print(f"\n🔄 Khởi động trình duyệt mới cho nguồn: {source['name']}...")
        driver = setup_driver()

        try:
            topic_folder = source['topic']
            topic_abbr = source['abbr']
            source_name = source['name']

            save_dir = os.path.join(DATA_ROOT_BASE, topic_folder)
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n{'=' * 60}")
            print(f"🚀 [{topic_abbr}] NGUỒN: {source_name}")

            # 1. Tìm STT
            current_idx = get_next_index(save_dir, topic_abbr, source_name)

            # 2. Lấy link (Dùng tab hiện tại)
            links = []
            if source['type'] == 'click':
                links = get_links_click_mode(driver, source)
            else:
                links = get_links_page_mode(driver, source)

            links = list(links)[:TARGET_PER_SOURCE]
            print(f"✅ Tìm thấy {len(links)} link. Đang xử lý...")

            # 3. Lưu file (MỞ TAB -> XỬ LÝ -> ĐÓNG TAB)
            saved_count = 0

            # Lưu lại handle của tab gốc (Tab chứa danh sách link)
            original_window = driver.current_window_handle

            for i, link in enumerate(links):
                try:
                    filename = f"{topic_abbr}_{source_name}_{current_idx}.txt"
                    filepath = os.path.join(save_dir, filename)

                    if os.path.exists(filepath):
                        current_idx += 1
                        continue

                    print(f"   [{i + 1}/{len(links)}] -> {filename}", end="\r")

                    content = ""

                    # Bước A: Thử dùng Newspaper3k trước (Nhẹ, không cần trình duyệt)
                    try:
                        article = Article(link)
                        article.download()
                        article.parse()
                        content = article.text.strip()
                    except:
                        pass

                    # Bước B: Nếu Newspaper thất bại, dùng Selenium Tab mới
                    if len(content) < 200:
                        # 1. Mở tab mới trắng tinh
                        driver.switch_to.new_window('tab')

                        # 2. Lấy nội dung
                        content = extract_content_selenium(driver, link, source['container'])

                        # 3. Đóng tab này ngay lập tức
                        driver.close()

                        # 4. Quay về tab gốc để đảm bảo driver không bị lạc
                        driver.switch_to.window(original_window)

                    if len(content) < 100: continue

                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"Url: {link}\n\n{content}")

                    saved_count += 1
                    current_idx += 1

                except Exception as e:
                    # Nếu có lỗi khi thao tác tab, đảm bảo quay về tab gốc
                    try:
                        if len(driver.window_handles) > 1:
                            driver.close()
                        driver.switch_to.window(original_window)
                    except:
                        pass

            print(f"\n🏁 Hoàn thành {source_name}: Đã lưu {saved_count} bài.")

        except Exception as e:
            print(f"Lỗi khi chạy nguồn {source['name']}: {e}")

        finally:
            # ✅ Đóng trình duyệt sau khi xong 1 nguồn để giải phóng hoàn toàn RAM
            print(f"🛑 Đóng trình duyệt của {source['name']}")
            driver.quit()

    print("\n🎉 TẤT CẢ HOÀN TẤT!")