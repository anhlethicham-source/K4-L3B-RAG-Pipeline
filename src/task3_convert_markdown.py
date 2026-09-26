"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.

-> Hoặc dùng công cụ nào bạn quen khác Markitdown

Các văn bản pháp luật trong corpus là bản scan (không có text layer), nên khi
MarkItDown trả về quá ít text, PDF được render từng trang và OCR bằng EasyOCR
(tiếng Việt). Kết quả OCR được cache bằng chính file .md đầu ra: chạy lại chỉ
OCR những file PDF mới hơn file Markdown tương ứng.
"""

import json
import re
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_MANIFEST = LANDING_DIR / "legal" / "sources.json"

# Số ký tự tối thiểu / trang để coi PDF là có text layer.
MIN_CHARS_PER_PAGE = 200
OCR_ZOOM = 2.0

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["vi"], gpu=False, verbose=False)
    return _ocr_reader


def ocr_pdf(path: Path) -> str:
    """Render từng trang PDF thành ảnh và OCR tiếng Việt."""
    import fitz

    reader = _get_ocr_reader()
    pages = []
    with fitz.open(path) as pdf:
        for number, page in enumerate(pdf, 1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_ZOOM, OCR_ZOOM))
            lines = reader.readtext(pixmap.tobytes("png"), detail=0, paragraph=True)
            text = "\n\n".join(line.strip() for line in lines if line.strip())
            pages.append(f"<!-- page {number} -->\n\n{text}")
            print(f"  OCR {path.name}: page {number}/{pdf.page_count}", flush=True)
    return "\n\n".join(pages)


# OCR hay đọc nhầm số 0/1 trong mã mẫu biểu thành O/I/L ("OI/CNKD" -> "01/CNKD").
_OCR_FORM_CODE = re.compile(r"(?<![\w/])([OIlL0-9]{1,2})(?=/[A-ZĐ0-9])")
_OCR_DIGITS = str.maketrans("OIlL", "0111")


def normalize_ocr_text(text: str) -> str:
    """Sửa lỗi OCR có quy luật trong mã văn bản/mẫu biểu, không đụng tới chữ thường."""
    return _OCR_FORM_CODE.sub(
        lambda m: m.group(1).translate(_OCR_DIGITS) if any(c.isdigit() for c in m.group(1))
        or len(m.group(1)) == 2 else m.group(1),
        text,
    )


def load_legal_manifest() -> dict:
    if LEGAL_MANIFEST.exists():
        return json.loads(LEGAL_MANIFEST.read_text(encoding="utf-8"))
    return {}


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào standardized/legal, OCR khi PDF là bản scan."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_legal_manifest()
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        # Chỉ convert tài liệu đã đăng ký nguồn trong manifest của Task 1.
        if manifest and path.name not in manifest:
            print(f"Skip (not in sources.json): {path.name}")
            continue

        target = output_dir / f"{path.stem}.md"
        if (
            target.exists()
            and target.stat().st_size > 0
            and target.stat().st_mtime >= path.stat().st_mtime
        ):
            # Không OCR lại; chỉ áp dụng chuẩn hoá (idempotent) cho bản đã có.
            content = target.read_text(encoding="utf-8")
            normalized = normalize_ocr_text(content)
            if normalized != content:
                target.write_text(normalized, encoding="utf-8")
            print(f"Up to date: {target.name}")
            continue

        text = converter.convert(str(path)).text_content or ""
        if path.suffix.lower() == ".pdf":
            import fitz

            with fitz.open(path) as pdf:
                page_count = pdf.page_count
            if len(text.strip()) < MIN_CHARS_PER_PAGE * page_count:
                print(f"Scanned PDF, running OCR: {path.name}")
                text = normalize_ocr_text(ocr_pdf(path))

        if not text.strip():
            print(f"Empty output, skipped: {path.name}")
            continue

        info = manifest.get(path.name, {})
        header = (
            f"# {info.get('title') or path.stem}\n\n"
            f"**Source:** {info.get('url') or path.name}\n\n---\n\n"
        )
        target.write_text(header + text.strip() + "\n", encoding="utf-8")
        print(f"Saved: {target.name}")


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MIN_BODY_LINE_CHARS = 40
# Dấu hiệu hết thân bài (footer/“bài liên quan”) của các site đã crawl.
END_MARKERS = (
    "Thêm VnExpress trên Google",
    "Nội dung này, đã nhận được",
    "Bài viết liên quan",
    "Tin liên quan",
)


# Một số site dùng chữ Cyrillic trông giống Latin (vd "TT-BTС"), làm BM25 lệch.
_HOMOGLYPHS = str.maketrans("АВСЕНКМОРТХасеорх", "ABCEHKMOPTXaceopx")


def clean_article_markdown(markdown: str) -> str:
    """Bỏ menu, nút chia sẻ, ảnh và các dòng chỉ gồm link do crawler giữ lại.

    Giữ lại dòng có đủ chữ sau khi bỏ URL, heading và dòng bảng.
    """
    markdown = markdown.translate(_HOMOGLYPHS)
    kept = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if kept and any(marker in stripped for marker in END_MARKERS):
            break
        text = _MD_LINK.sub(r"\1", _MD_IMAGE.sub("", stripped))
        text = re.sub(r"[*_#>|\-\s]+", " ", text).strip()
        is_heading = stripped.startswith("#") and len(text) >= 10
        is_table = stripped.startswith("|") and len(text) >= 3
        if is_heading or is_table or len(text) >= MIN_BODY_LINE_CHARS:
            kept.append(_MD_LINK.sub(r"\1", _MD_IMAGE.sub("", line)).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def convert_news_articles() -> None:
    """Convert JSON vào standardized/news, giữ metadata ở đầu file."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not str(data.get("content_markdown", "")).strip():
            print(f"Empty article, skipped: {path.name}")
            continue
        header = (
            f"# {data['title'].translate(_HOMOGLYPHS)}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + clean_article_markdown(data["content_markdown"]) + "\n",
            encoding="utf-8",
        )


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
