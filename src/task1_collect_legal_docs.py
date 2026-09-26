"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.

Lưu ý: trang vbpl.vn/van-ban/chi-tiet/... là trang HTML render bằng JavaScript,
không phải file PDF. Với các nguồn này, tải file thủ công vào DATA_DIR rồi
ghi URL trang gốc vào SOURCES (url=None cho download_url) để giữ nguồn trích dẫn.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "sources.json"
ALLOWED_SUFFIXES = {".pdf", ".doc", ".docx"}
MIN_DOCUMENTS = 3

# filename (không dấu) -> nguồn.
#   url:          trang công khai dùng để trích dẫn.
#   download_url: link tải trực tiếp PDF/DOCX; None nếu đã tải thủ công.
SOURCES: dict[str, dict] = {
    "tt-18-2006-btc.pdf": {
        "title": "Thông tư 18/2026/TT-BTC",
        "url": None,  # TODO: điền URL trang gốc
        "download_url": None,
    },
    "vanbangoc-180-ttg-38426.pdf": {
        "title": "Quyết định 180/TTg",
        "url": None,  # TODO: điền URL trang gốc
        "download_url": None,
    },
    "vanbangoc-73-2016-nd-cp.pdf": {
        "title": "Nghị định 73/2016/NĐ-CP",
        "url": None,  # TODO: điền URL trang gốc
        "download_url": None,
    },
    "vanbangoc-tt-152-2025-btc.pdf": {
        "title": "Thông tư 152/2025/TT-BTC",
        "url": None,  # TODO: điền URL trang gốc
        "download_url": None,
    },
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def slugify_filename(name: str) -> str:
    """Chuyển tên file về dạng không dấu, chữ thường, nối bằng '-'."""
    path = Path(name)
    stem = path.stem.replace("đ", "d").replace("Đ", "D")
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", stem).strip("-").lower()
    return f"{stem}{path.suffix.lower()}"


def is_valid_document(content: bytes, suffix: str) -> bool:
    """Kiểm tra magic bytes để tránh lưu trang HTML thành .pdf/.docx."""
    if suffix == ".pdf":
        return content.startswith(b"%PDF-")
    if suffix == ".docx":
        return content.startswith(b"PK")
    if suffix == ".doc":
        return content.startswith(b"\xd0\xcf\x11\xe0")
    return False


def normalize_existing_files() -> None:
    """Đổi tên các file tải thủ công sang tên không dấu."""
    for path in DATA_DIR.iterdir():
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        target = DATA_DIR / slugify_filename(path.name)
        if target.name != path.name and not target.exists():
            path.rename(target)
            print(f"Renamed: {path.name} -> {target.name}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    normalize_existing_files()

    headers = {"User-Agent": "Mozilla/5.0 (educational RAG lab)"}
    for filename, source in SOURCES.items():
        target = DATA_DIR / filename
        if target.exists():
            print(f"Exists:  {filename}")
            continue
        if not source.get("download_url"):
            print(f"Missing: {filename} (không có download_url, hãy tải thủ công)")
            continue

        response = requests.get(source["download_url"], headers=headers, timeout=30)
        response.raise_for_status()
        if not is_valid_document(response.content, target.suffix.lower()):
            print(f"Skipped: {filename} (nội dung tải về không phải PDF/DOCX)")
            continue
        target.write_bytes(response.content)
        print(f"Saved:   {filename} ({len(response.content):,} bytes)")

    documents = [
        path
        for path in sorted(DATA_DIR.iterdir())
        if path.suffix.lower() in ALLOWED_SUFFIXES
        and is_valid_document(path.read_bytes()[:8], path.suffix.lower())
    ]
    manifest = {
        path.name: {
            "title": SOURCES.get(path.name, {}).get("title", path.stem),
            "url": SOURCES.get(path.name, {}).get("url"),
        }
        for path in documents
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifest: {MANIFEST_PATH.name} ({len(manifest)} documents)")

    if len(documents) < MIN_DOCUMENTS:
        raise RuntimeError(
            f"Cần ít nhất {MIN_DOCUMENTS} tài liệu PDF/DOCX hợp lệ, hiện có {len(documents)}"
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    setup_directory()
    download_documents()
