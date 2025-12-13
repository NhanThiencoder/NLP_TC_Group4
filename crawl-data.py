import os
import time
import random
import re
import requests

from bs4 import BeautifulSoup
from newspaper import Article
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# =========================================================
# 1. CẤU HÌNH THƯ MỤC / TOPIC / SOURCE
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT_FOLDER = os.path.join(BASE_DIR, "Crawled_Data")
print("SAVE TO FOLDER:", DATA_ROOT_FOLDER)

TOPICS = {
    # Du lịch – target 2700
    "DuLich": {
        "abbr": "DL",
        "target_total": 2700,
        "tolerance_max": 300,
        "folder": os.path.join(DATA_ROOT_FOLDER, "Du lịch"),
    },
    # Chứng khoán – target 2600
    "ChungKhoan": {
        "abbr": "CK",
        "target_total": 2600,
        "tolerance_max": 300,
        "folder": os.path.join(DATA_ROOT_FOLDER, "Chứng khoán"),
    },
    # Bất động sản – target 2800
    "BatDongSan": {
        "abbr": "BĐS",
        "target_total": 2800,
        "tolerance_max": 300,
        "folder": os.path.join(DATA_ROOT_FOLDER, "Bất động sản"),
    },
    # Ẩm thực – target 2900
    "AmThuc": {
        "abbr": "AT",
        "target_total": 2900,
        "tolerance_max": 300,
        "folder": os.path.join(DATA_ROOT_FOLDER, "Ẩm thực"),
    },
}

SOURCES = {
    # Du lịch: Lao Động + Zing + VOV + VTV
    "DuLich": [
        {"source": "LD",   "url": "https://laodong.vn/du-lich"},
        {"source": "ZING", "url": "https://znews.vn/du-lich.html"},
        {"source": "VOV",  "url": "https://vov.vn/du-lich"},
        {"source": "VTV",  "url": "https://vtv.vn/doi-song/du-lich.htm"},
    ],
    # Chứng khoán: chỉ Lao Động
    "ChungKhoan": [
        {"source": "LD", "url": "https://laodong.vn/kinh-te/chung-khoan"},
        {"source": "LD", "url": "https://laodong.vn/tags/chung-khoan-31135.ldo"},
    ],
    # BĐS: chỉ Lao Động
    "BatDongSan": [
        {"source": "LD", "url": "https://laodong.vn/bat-dong-san"},
        {"source": "LD", "url": "https://laodong.vn/tags/thi-truong-bat-dong-san-4494.ldo"},
        {"source": "LD", "url": "https://laodong.vn/tags/bat-dong-san-nha-o-430069.ldo"},
    ],
    # Ẩm thực: TT, TN, NLD, VNE, VNN, LD
    "AmThuc": [
        # Tuổi Trẻ
        {"source": "TT",  "url": "https://tuoitre.vn/van-hoa/am-thuc.htm"},
        # Thanh Niên
        {"source": "TN",  "url": "https://thanhnien.vn/doi-song/am-thuc.htm"},
        # Người Lao Động
        {"source": "NLD", "url": "https://nld.com.vn/du-lich-xanh/am-thuc.htm"},
        # VnExpress – ẩm thực / cooking
        {"source": "VNE", "url": "https://vnexpress.net/du-lich/am-thuc"},
        {"source": "VNE", "url": "https://vnexpress.net/doi-song/noi-tro/food"},
        {"source": "VNE", "url": "https://vnexpress.net/doi-song/cooking/mon-an"},
        {"source": "VNE", "url": "https://vnexpress.net/doi-song/cooking/thuc-don"},
        # Vietnamnet
        {"source": "VNN", "url": "https://vietnamnet.vn/doi-song/am-thuc"},
        {"source": "VNN", "url": "https://vietnamnet.vn/mon-ngon-moi-ngay-tag14888584744236181834.html"},
        # Lao Động – ẩm thực
        {"source": "LD",  "url": "https://laodong.vn/du-lich/am-thuc"},
        {"source": "LD",  "url": "https://laodong.vn/tags/am-thuc-5075.ldo"},
    ],
}

MAX_PAGES_PER_SOURCE = 80
REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = (1.0, 3.0)


# =========================================================
# 2. HÀM HỖ TRỢ
# =========================================================
def ensure_folder(folder: str):
    os.makedirs(folder, exist_ok=True)


def normalize_url(base_url: str, href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    from urllib.parse import urljoin
    return urljoin(base_url, href)


def count_existing_articles(folder: str, abbr: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(
        1
        for f in os.listdir(folder)
        if f.lower().endswith(".txt") and f.startswith(f"{abbr}_")
    )


def get_page_url(base_url: str, source: str, page: int) -> str:
    if page == 1:
        return base_url

    if source == "LD":
        return f"{base_url}?page={page}"

    if source in {"TT", "TN", "VNN"}:
        return f"{base_url}?page={page}"

    if source == "VNE":
        return f"{base_url}-p{page}"

    if source in {"ZING", "VOV", "VTV"}:
        return f"{base_url}?page={page}"

    return base_url


def fetch_html_requests(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.text
        print(f"    + HTTP {resp.status_code} khi GET {url}")
        return ""
    except Exception as e:
        print(f"    + Lỗi requests tới {url}: {e}")
        return ""


def get_rendered_html(url: str) -> str:
    """Dùng Playwright để render trang (cho Lao Động, dùng domcontentloaded)."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
            except PlaywrightTimeoutError as e:
                print(f"    + Timeout Playwright khi goto {url}: {e}. Dùng content hiện tại.")
            page.wait_for_timeout(4000)
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        print(f"    + Lỗi Playwright khi load {url}: {e}")
        return ""


# =========================================================
# 3. LẤY LINK BÀI
# =========================================================
def collect_links_for_source(topic_key: str,
                             topic_cfg: dict,
                             source_cfg: dict,
                             need: int,
                             global_seen: set):
    base_url = source_cfg["url"]
    source = source_cfg["source"]
    links = []
    page = 1

    print(f"\n>>> CRAWL topic={topic_key}, source={source}, cần thêm ~{need} link")
    while len(links) < need and page <= MAX_PAGES_PER_SOURCE:
        page_url = get_page_url(base_url, source, page)
        print(f"  - {topic_key}/{source}: page {page} -> {page_url}")

        if source == "LD":
            html = get_rendered_html(page_url)
        else:
            html = fetch_html_requests(page_url)

        if not html:
            print("    + Không lấy được HTML, dừng source.")
            break

        soup = BeautifulSoup(html, "html.parser")
        article_links = []

        # ---------- Lao Động ----------
        if source == "LD":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "laodong.vn" not in href:
                    continue
                if "/video/" in href:
                    continue
                if not re.search(r"-\d+\.(?:ldo|html)$", href):
                    continue

                if topic_key == "DuLich":
                    if "du-lich" not in href:
                        continue
                elif topic_key == "ChungKhoan":
                    if "chung-khoan" not in href:
                        continue
                elif topic_key == "BatDongSan":
                    if "bat-dong-san" not in href:
                        continue
                elif topic_key == "AmThuc":
                    if "am-thuc" not in href and "du-lich" not in href:
                        continue

                article_links.append(href)

        # ---------- Tuổi Trẻ ----------
        elif source == "TT":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "tuoitre.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not re.search(r"-\d+\.htm$", href):
                    continue
                article_links.append(href)

        # ---------- Thanh Niên ----------
        elif source == "TN":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "thanhnien.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not re.search(r"-\d+\.html$", href):
                    continue
                article_links.append(href)

        # ---------- Người Lao Động (ẩm thực NLD) ----------
        elif source == "NLD":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "nld.com.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not href.endswith(".htm"):
                    continue
                article_links.append(href)

        # ---------- VnExpress ----------
        elif source == "VNE":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "vnexpress.net" not in href:
                    continue
                if "/video/" in href:
                    continue
                if not re.search(r"-\d+\.html$", href):
                    continue
                article_links.append(href)

        # ---------- Vietnamnet ----------
        elif source == "VNN":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "vietnamnet.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not re.search(r"-\d+\.html$", href):
                    continue
                article_links.append(href)

        # ---------- Zing ----------
        elif source == "ZING":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "znews.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not re.search(r"-post\d+\.html$", href):
                    continue
                if topic_key == "DuLich" and "du-lich" not in href:
                    continue
                article_links.append(href)

        # ---------- VOV ----------
        elif source == "VOV":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "vov.vn" not in href:
                    continue
                if "video" in href or "podcast" in href:
                    continue
                if not href.endswith(".vov"):
                    continue
                if topic_key == "DuLich" and "/du-lich" not in href:
                    continue
                article_links.append(href)

        # ---------- VTV ----------
        elif source == "VTV":
            for a in soup.select("a[href]"):
                href = normalize_url(base_url, a.get("href"))
                if not href:
                    continue
                if "vtv.vn" not in href:
                    continue
                if "video" in href:
                    continue
                if not href.endswith(".htm"):
                    continue
                if topic_key == "DuLich" and "/du-lich" not in href:
                    continue
                article_links.append(href)

        if not article_links:
            print(f"    + Không tìm được link bài nào trên trang {page}.")
            break

        new_this_page = 0
        for link in article_links:
            if not link:
                continue
            if "podcast" in link or len(link) < 15:
                continue
            if link in global_seen:
                continue
            global_seen.add(link)
            links.append(link)
            new_this_page += 1
            if len(links) >= need:
                break

        print(f"    + Link mới trang {page}: {new_this_page} (tổng {len(links)}/{need})")

        if new_this_page == 0:
            break

        page += 1
        time.sleep(random.uniform(*SLEEP_BETWEEN_REQUESTS))

    if not links:
        print("    + Không thu được link nào, bỏ qua nguồn này.")
    return links


# =========================================================
# 4. EXTRACT NỘI DUNG BÁO
# =========================================================
def extract_ld_article_text(url: str) -> str:
    """Lấy nội dung Lao Động bằng Playwright + BeautifulSoup."""
    html = get_rendered_html(url)
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    candidates = []
    candidates.extend(soup.select("article"))
    candidates.extend(soup.select('div[class*="content"]'))
    candidates.extend(soup.select('div[class*="article"]'))
    candidates.extend(soup.select('div[class*="detail"]'))
    candidates.extend(soup.select('div[class*="body"]'))

    root = candidates[0] if candidates else soup

    paragraphs = []
    for p in root.find_all("p"):
        txt = p.get_text(strip=True)
        if not txt or len(txt) < 10:
            continue
        paragraphs.append(txt)

    return "\n".join(paragraphs).strip()


def extract_article_text_generic(url: str) -> str:
    """Ưu tiên newspaper3k, fallback BeautifulSoup."""
    try:
        art = Article(url)
        art.download()
        art.parse()
        text = ((art.title or "") + "\n\n" + (art.text or "")).strip()
        if text and len(text) > 200:
            return text
    except Exception:
        pass

    html = fetch_html_requests(url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    candidates = []
    for div in soup.find_all(["article", "div"]):
        txt = div.get_text(" ", strip=True)
        if len(txt) > 300:
            candidates.append(div)

    root = candidates[0] if candidates else soup

    paragraphs = []
    for p in root.find_all("p"):
        txt = p.get_text(strip=True)
        if not txt or len(txt) < 10:
            continue
        paragraphs.append(txt)

    return "\n".join(paragraphs).strip()


def extract_article_text(url: str, source: str) -> str:
    if source == "LD":
        return extract_ld_article_text(url)
    return extract_article_text_generic(url)


# =========================================================
# 5. LƯU NỘI DUNG BÀI
# =========================================================
def save_articles(topic_cfg: dict, source_cfg: dict, links: list):
    folder = topic_cfg["folder"]
    abbr = topic_cfg["abbr"]
    source = source_cfg["source"]

    ensure_folder(folder)

    existing_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(".txt") and f.startswith(f"{abbr}_{source}_")
    ]
    next_index = len(existing_files) + 1

    saved = 0
    idx = next_index

    print(f"    + Lưu vào {folder}, source={source}, bắt đầu từ #{idx}")
    for url in links:
        try:
            text = extract_article_text(url, source)
            if not text or len(text) < 200:
                continue

            filename = f"{abbr}_{source}_{idx:05d}.txt"
            filepath = os.path.join(folder, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(url + "\n\n" + text)

            saved += 1
            idx += 1
            print(".", end="", flush=True)
            time.sleep(random.uniform(0.3, 1.0))
        except Exception as e:
            print(f"\n    + Lỗi khi lưu bài {url}: {e}")
            continue

    print(f"\n    + Đã lưu {saved} bài từ source={source}")
    return saved


# =========================================================
# 6. MAIN
# =========================================================
def main():
    print(f"DATA_ROOT_FOLDER = {DATA_ROOT_FOLDER}")

    total_new_all_topics = 0

    for topic_key, topic_cfg in TOPICS.items():
        folder = topic_cfg["folder"]
        abbr = topic_cfg["abbr"]
        target_total = topic_cfg["target_total"]
        tolerance_max = topic_cfg["tolerance_max"]

        ensure_folder(folder)

        current = count_existing_articles(folder, abbr)
        remaining = max(0, target_total - current)

        print("\n" + "=" * 60)
        print(f">>> CHỦ ĐỀ: {topic_key} ({abbr})")
        print(f"  - Target: {target_total}, cho phép thiếu ≤ {tolerance_max}")
        print(f"  - Đang có: {current} bài trong folder: {folder}")
        print(f"  - Còn thiếu để đạt target: {remaining} bài")

        if remaining <= 0:
            print("  -> ĐÃ ĐỦ HOẶC VƯỢT TARGET, BỎ QUA.")
            continue

        total_saved_topic = 0
        global_seen = set()

        for source_cfg in SOURCES.get(topic_key, []):
            if remaining <= 0:
                break

            source = source_cfg["source"]
            print(f"\n>>> XỬ LÝ SOURCE = {source}  (còn cần ~{remaining} bài để đạt target)")

            links = collect_links_for_source(topic_key, topic_cfg, source_cfg, remaining, global_seen)
            if not links:
                continue

            saved = save_articles(topic_cfg, source_cfg, links)
            total_saved_topic += saved
            remaining = max(0, target_total - (current + total_saved_topic))

            print(f">>> Sau source {source}: còn cần ~{remaining} bài để đạt target")

        final_total = current + total_saved_topic
        final_missing = max(0, target_total - final_total)

        print(f"\n>>> KẾT THÚC CHỦ ĐỀ {topic_key}:")
        print(f"    - Trước khi crawl: {current} bài")
        print(f"    - Sau khi crawl:  {final_total} bài (target {target_total})")
        print(f"    - Thiếu còn lại:  {final_missing} bài")

        if final_missing == 0:
            print("    -> ĐÃ ĐỦ TARGET.")
        elif final_missing <= tolerance_max:
            print(f"    -> CHƯA ĐỦ NHƯNG THIẾU TRONG NGƯỠNG CHO PHÉP (≤ {tolerance_max}).")
        else:
            print(f"    -> THIẾU QUÁ NHIỀU (> {tolerance_max}), CÓ THỂ WEB KHÔNG CÓ ĐỦ BÀI.")

        total_new_all_topics += total_saved_topic

    print("\n" + "=" * 60)
    print(f"TỔNG CỘNG: thêm mới {total_new_all_topics} bài cho tất cả chủ đề.")


if __name__ == "__main__":
    main()
