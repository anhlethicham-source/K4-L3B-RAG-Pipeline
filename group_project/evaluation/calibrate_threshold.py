"""
Calibrate SCORE_THRESHOLD cho fallback bằng best dense cosine score.

So sánh phân bố score của query in-domain (golden dataset) với query ngoài domain.
Threshold đề xuất = trung điểm giữa min(in-domain) và max(out-of-domain).

    python -m group_project.evaluation.calibrate_threshold
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.task9_retrieval_pipeline import best_dense_score  # noqa: E402

OUT_OF_DOMAIN = [
    "Công thức nấu phở bò Hà Nội gồm những nguyên liệu gì?",
    "Đội tuyển Việt Nam vô địch AFF Cup năm nào?",
    "Làm thế nào để cài đặt Python trên Windows?",
    "Thời tiết Đà Lạt tháng 12 thế nào?",
    "Ai là tác giả Truyện Kiều?",
    "Cách chăm sóc cây xương rồng trong nhà",
    "Giá vé máy bay Hà Nội đi Tokyo bao nhiêu?",
    "What is the capital of Australia?",
]


def main() -> None:
    golden = json.loads((ROOT / "group_project/evaluation/golden_dataset.json").read_text(encoding="utf-8"))
    in_domain = [(item["question"], best_dense_score(item["question"])) for item in golden]
    out_domain = [(q, best_dense_score(q)) for q in OUT_OF_DOMAIN]

    for label, rows in (("IN-DOMAIN", in_domain), ("OUT-OF-DOMAIN", out_domain)):
        print(f"\n{label}")
        for question, score in sorted(rows, key=lambda r: r[1]):
            print(f"  {score:.3f}  {question[:70]}")

    low_in = min(s for _, s in in_domain)
    high_out = max(s for _, s in out_domain)
    print(f"\nmin in-domain = {low_in:.3f} · max out-of-domain = {high_out:.3f}")
    print(f"Suggested SCORE_THRESHOLD = {(low_in + high_out) / 2:.2f}")
    if low_in <= high_out:
        print("Cảnh báo: hai phân bố chồng lấn, threshold sẽ có false positive/negative.")


if __name__ == "__main__":
    main()
