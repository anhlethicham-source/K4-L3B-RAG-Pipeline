"""
Evaluation A/B: Config A (dense-only) vs Config B (hybrid dense + BM25 → RRF).

Hai config dùng chung golden dataset, generator, judge, prompt và top_k; chỉ
khác retrieval strategy (retrieve(..., use_reranking=False/True)).

Metrics (định nghĩa theo RAGAS, chấm bằng LLM-as-judge trong một call/mẫu để
vừa quota free tier của Gemini):
    - faithfulness:       tỉ lệ claim trong answer được context hỗ trợ.
    - answer_relevance:   mức answer trả lời đúng trọng tâm câu hỏi (0–1).
    - context_recall:     tỉ lệ ý trong expected_answer tìm thấy trong context.
    - context_precision:  average precision của các context liên quan theo rank.
Ngoài ra đo không cần LLM: hit@k theo expected_source và latency retrieval.

Chạy:
    python -m group_project.evaluation.run_evaluation            # cả 2 config
    python -m group_project.evaluation.run_evaluation --retrieval-only
Kết quả từng mẫu được cache vào results_<config>.jsonl nên chạy lại sẽ tiếp tục.
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.task10_generation import (  # noqa: E402
    LLM_MODEL,
    LLM_PROVIDER,
    REFUSAL,
    call_llm,
    generate_from_chunks,
)
from src.task9_retrieval_pipeline import retrieve  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN = EVAL_DIR / "golden_dataset.json"
TOP_K = 5
CONFIGS = {"A_dense": False, "B_hybrid": True}
METRICS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]
CALL_INTERVAL_S = 1.0  # giãn cách call để không vượt rate limit của provider

JUDGE_PROMPT = """Bạn là giám khảo đánh giá hệ thống RAG. Chỉ trả về JSON hợp lệ, không giải thích thêm.

Question: {question}

Expected answer (ground truth): {expected_answer}

Retrieved contexts:
{contexts}

Generated answer: {answer}

Hãy chấm:
1. "claims": tách Generated answer thành các khẳng định ngắn; mỗi phần tử {{"claim": str, "supported": bool}} với supported=true nếu khẳng định suy ra được từ Retrieved contexts. Nếu answer là lời từ chối, trả [].
2. "answer_relevance": số 0..1, answer trả lời đúng và đầy đủ trọng tâm câu hỏi đến mức nào (lời từ chối = 0).
3. "gt_statements": tách Expected answer thành các ý; mỗi phần tử {{"statement": str, "in_context": bool}} với in_context=true nếu ý đó có trong Retrieved contexts.
4. "context_relevant": list bool theo đúng thứ tự contexts, true nếu context đó hữu ích để trả lời câu hỏi.

JSON schema: {{"claims": [...], "answer_relevance": float, "gt_statements": [...], "context_relevant": [...]}}"""


def llm_with_retry(system: str, prompt: str, attempts: int = 5) -> str:
    delay = 15.0
    for attempt in range(attempts):
        try:
            time.sleep(CALL_INTERVAL_S)
            return call_llm(system, prompt)
        except Exception as error:  # 429 / 503 của provider
            if attempt == attempts - 1:
                raise
            print(f"    LLM error ({error!r:.120}), retry in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    return ""


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def average_precision(relevant: list[bool]) -> float:
    hits, total = 0, 0.0
    for rank, flag in enumerate(relevant, 1):
        if flag:
            hits += 1
            total += hits / rank
    return total / hits if hits else 0.0


def judge(question: str, expected: str, chunks: list[dict], answer: str) -> dict:
    contexts = "\n\n".join(
        f"[{i}] ({c['metadata']['source']}) {c['content']}" for i, c in enumerate(chunks, 1)
    ) or "(không có)"
    raw = llm_with_retry(
        "Bạn là giám khảo nghiêm khắc, trả lời bằng JSON.",
        JUDGE_PROMPT.format(
            question=question, expected_answer=expected, contexts=contexts, answer=answer
        ),
    )
    data = parse_json(raw)
    claims = data.get("claims") or []
    statements = data.get("gt_statements") or []
    relevant = [bool(x) for x in (data.get("context_relevant") or [])][: len(chunks)]
    is_refusal = answer.strip() == REFUSAL
    return {
        # Lời từ chối không chứa claim sai -> faithful; RAGAS cũng không phạt refusal.
        "faithfulness": 1.0 if is_refusal or not claims
        else sum(bool(c.get("supported")) for c in claims) / len(claims),
        "answer_relevance": 0.0 if is_refusal else float(data.get("answer_relevance") or 0.0),
        "context_recall": sum(bool(s.get("in_context")) for s in statements) / len(statements)
        if statements else 0.0,
        "context_precision": average_precision(relevant),
        "judge_raw": data,
    }


def hit_at_k(chunks: list[dict], expected_sources: list[str]) -> float:
    return float(any(c["metadata"]["source"] in expected_sources for c in chunks))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def context_coverage(chunk: dict, expected_context: str) -> float:
    """Tỉ lệ token của expected_context xuất hiện trong chunk (không cần LLM)."""
    expected = _tokens(expected_context)
    return len(expected & _tokens(chunk["content"])) / len(expected) if expected else 0.0


def evidence_rank(chunks: list[dict], expected_context: str, min_coverage: float = 0.6) -> int | None:
    """Rank (1-based) của chunk đầu tiên chứa phần lớn expected_context."""
    for rank, chunk in enumerate(chunks, 1):
        if context_coverage(chunk, expected_context) >= min_coverage:
            return rank
    return None


def run_config(name: str, use_reranking: bool, dataset: list[dict], retrieval_only: bool) -> list[dict]:
    out_path = EVAL_DIR / f"results_{name}.jsonl"
    done = {}
    if out_path.exists() and not retrieval_only:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row

    rows = []
    for item in dataset:
        if item["id"] in done:
            rows.append(done[item["id"]])
            continue
        start = time.perf_counter()
        chunks = retrieve(item["question"], top_k=TOP_K, use_reranking=use_reranking)
        latency = time.perf_counter() - start
        rank = evidence_rank(chunks, item["expected_context"])
        row = {
            "id": item["id"],
            "question": item["question"],
            "config": name,
            "retrieved": [c["id"] for c in chunks],
            "retrieval_methods": sorted({c["retrieval_method"] for c in chunks}),
            "hit_at_k": hit_at_k(chunks, item.get("expected_source", [])),
            "evidence_hit": float(rank is not None),
            "evidence_mrr": 1.0 / rank if rank else 0.0,
            "retrieval_latency_s": round(latency, 3),
        }
        if not retrieval_only:
            time.sleep(CALL_INTERVAL_S)
            result = generate_from_chunks(item["question"], chunks)
            row["answer"] = result["answer"]
            row.update(judge(item["question"], item["expected_answer"], chunks, result["answer"]))
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        metrics = " ".join(f"{m[:5]}={row[m]:.2f}" for m in METRICS if m in row)
        print(f"  [{name}] {item['id']} hit={row['hit_at_k']:.0f} "
              f"evidence_rank={rank} {metrics}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    keys = ["hit_at_k", "evidence_hit", "evidence_mrr", "retrieval_latency_s"] + [m for m in METRICS if m in rows[0]]
    summary = {key: statistics.mean(row[key] for row in rows) for key in keys}
    if all(m in summary for m in METRICS):
        summary["average"] = statistics.mean(summary[m] for m in METRICS)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Golden: {len(dataset)} câu · generator/judge: {LLM_PROVIDER}/{LLM_MODEL} · top_k={TOP_K}")
    retrieve("warm-up", top_k=1)  # load embedding model trước khi đo latency
    summaries = {}
    for name, use_reranking in CONFIGS.items():
        rows = run_config(name, use_reranking, dataset, args.retrieval_only)
        summaries[name] = summarize(rows)

    print("\nMetric                 A_dense   B_hybrid   Delta")
    for key in summaries["A_dense"]:
        a, b = summaries["A_dense"][key], summaries["B_hybrid"][key]
        print(f"{key:<22} {a:>8.3f} {b:>10.3f} {b - a:>+8.3f}")
    suffix = "retrieval" if args.retrieval_only else "full"
    (EVAL_DIR / f"summary_{suffix}.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
