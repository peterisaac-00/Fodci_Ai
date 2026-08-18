from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any

import torch

from backend_ai.model import FodciModel, ModelConfig
from backend_ai.tokenizer import EOS_ID, FodciTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "training_data" / "fundamentals" / "evaluation" / "stage_01.jsonl"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "fodci-tiny-v1.pt"
DEFAULT_REPORT = PROJECT_ROOT / "artifacts" / "evaluation" / "stage1_baseline.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "artifacts" / "evaluation" / "stage1_baseline.md"
REQUIRED_FIELDS = {
    "benchmark_id",
    "version",
    "split",
    "category",
    "question",
    "expected_answer",
    "required_keywords",
    "minimum_keyword_coverage",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark dataset not found: {path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on benchmark line {line_number}") from exc
        if set(record) != REQUIRED_FIELDS:
            raise ValueError(f"benchmark line {line_number} has an invalid schema")
        if record["split"] != "benchmark":
            raise ValueError(f"benchmark line {line_number} is not held out")
        if not isinstance(record["required_keywords"], list) or not record["required_keywords"]:
            raise ValueError(f"benchmark line {line_number} requires keywords")
        if record["benchmark_id"] in seen_ids or record["question"] in seen_questions:
            raise ValueError(f"duplicate benchmark identity on line {line_number}")
        seen_ids.add(record["benchmark_id"])
        seen_questions.add(record["question"])
        records.append(record)
    if not records:
        raise ValueError("benchmark dataset is empty")
    return records


def load_model(checkpoint_path: Path, model_config: ModelConfig) -> tuple[FodciModel, int]:
    model = FodciModel(model_config)
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"checkpoint does not contain a state dictionary: {checkpoint_path}")
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return model, parameter_count


def generate_greedy(model: FodciModel, tokenizer: FodciTokenizer, question: str, *, max_new_tokens: int) -> tuple[str, int]:
    prompt = f"### Instruction\n{question}\n\n### Input\nNone\n\n### Response\n"
    prompt_ids = tokenizer.encode(prompt)
    context_limit = int(getattr(model.config, "context_length", 256))
    prompt_ids = prompt_ids[-max(1, context_limit - max_new_tokens) :]
    generated = torch.tensor([prompt_ids], dtype=torch.long)
    produced = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated[:, -context_limit:])
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated = torch.cat([generated, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            produced += 1
            if next_token == EOS_ID:
                break
    response_ids = generated[0].tolist()[len(prompt_ids) :]
    return tokenizer.decode(response_ids).strip(), produced


def score_response(record: dict[str, Any], response: str) -> dict[str, Any]:
    normalized = normalize_text(response)
    keywords = [str(keyword) for keyword in record["required_keywords"]]
    matched = [keyword for keyword in keywords if normalize_text(keyword) in normalized]
    coverage = len(matched) / len(keywords)
    threshold = float(record["minimum_keyword_coverage"])
    return {
        "matched_keywords": matched,
        "missing_keywords": [keyword for keyword in keywords if keyword not in matched],
        "keyword_coverage": round(coverage, 6),
        "minimum_keyword_coverage": threshold,
        "non_empty": bool(normalized),
        "passed": bool(normalized) and coverage >= threshold,
    }


def evaluate(records: list[dict[str, Any]], model: FodciModel, tokenizer: FodciTokenizer, *, max_new_tokens: int) -> dict[str, Any]:
    task_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in records:
        task_started = time.perf_counter()
        response, generated_tokens = generate_greedy(model, tokenizer, record["question"], max_new_tokens=max_new_tokens)
        score = score_response(record, response)
        task_results.append({
            "benchmark_id": record["benchmark_id"],
            "category": record["category"],
            "question": record["question"],
            "response": response,
            "generated_tokens": generated_tokens,
            "duration_seconds": round(time.perf_counter() - task_started, 6),
            **score,
        })
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in task_results:
        by_category[result["category"]].append(result)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(items)
        return {
            "items": count,
            "passed": sum(item["passed"] for item in items),
            "pass_rate": round(sum(item["passed"] for item in items) / count, 6) if count else None,
            "non_empty_rate": round(sum(item["non_empty"] for item in items) / count, 6) if count else None,
            "average_keyword_coverage": round(sum(item["keyword_coverage"] for item in items) / count, 6) if count else None,
            "average_generated_tokens": round(sum(item["generated_tokens"] for item in items) / count, 6) if count else None,
        }

    return {
        "aggregate": aggregate(task_results),
        "by_category": {category: aggregate(items) for category, items in sorted(by_category.items())},
        "tasks": task_results,
        "duration_seconds": round(time.perf_counter() - started, 6),
    }


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["evaluation"]["aggregate"]
    lines = [
        "# Fodci Stage 1 Baseline Evaluation",
        "",
        "> This report is a pre-training baseline for the approximately 11M-parameter Fodci model. It must be preserved before comparing later checkpoints.",
        "",
        "## Run Identity",
        "",
        f"- **Run ID:** `{report['run_id']}`",
        f"- **Model:** `{report['model']['version']}`",
        f"- **Parameters:** `{report['model']['parameter_count']:,}`",
        f"- **Checkpoint:** `{report['model']['checkpoint']}`",
        f"- **Dataset:** `{report['dataset']['path']}`",
        f"- **Dataset records:** `{report['dataset']['records']}`",
        f"- **Seed:** `{report['protocol']['seed']}`",
        f"- **Maximum generated tokens:** `{report['protocol']['max_new_tokens']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Passed tasks | {aggregate['passed']} / {aggregate['items']} |",
        f"| Keyword pass rate | {aggregate['pass_rate']:.2%} |",
        f"| Non-empty response rate | {aggregate['non_empty_rate']:.2%} |",
        f"| Average keyword coverage | {aggregate['average_keyword_coverage']:.2%} |",
        f"| Average generated tokens | {aggregate['average_generated_tokens']:.2f} |",
        "",
        "## Category Metrics",
        "",
        "| Category | Items | Pass rate | Non-empty rate | Keyword coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, metrics in report["evaluation"]["by_category"].items():
        lines.append(f"| {category} | {metrics['items']} | {metrics['pass_rate']:.2%} | {metrics['non_empty_rate']:.2%} | {metrics['average_keyword_coverage']:.2%} |")
    lines.extend(["", "## Interpretation", "", "The keyword score is a deterministic proxy for Stage 1 concept coverage; it is not a substitute for human review or a semantic judge. Future checkpoints must use the same dataset, prompt template, seed, decoding rule, and scoring thresholds for a valid comparison.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the held-out Stage 1 Fodci baseline benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-version", default="fodci-tiny-v1")
    parser.add_argument("--run-prefix", default="stage1-baseline")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.seed < 0 or args.max_new_tokens <= 0:
        raise ValueError("seed must be non-negative and max-new-tokens must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_benchmark(args.dataset)
    model_config = ModelConfig(seed=args.seed)
    model, parameter_count = load_model(args.checkpoint, model_config)
    tokenizer = FodciTokenizer()
    evaluation = evaluate(records, model, tokenizer, max_new_tokens=args.max_new_tokens)
    dataset_fingerprint = sha256_file(args.dataset)
    checkpoint_fingerprint = sha256_file(args.checkpoint)
    identity = {"version": args.model_version, "parameter_count": parameter_count, "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_fingerprint}
    protocol = {"seed": args.seed, "decoding": "greedy_argmax", "temperature": 0.0, "max_new_tokens": args.max_new_tokens, "prompt_template": "instruction-input-response-v1", "device": "cpu"}
    run_id = args.run_prefix + "-" + hashlib.sha256(canonical_json({"model": identity, "dataset_sha256": dataset_fingerprint, "protocol": protocol}).encode("utf-8")).hexdigest()[:16]
    report = {"format": "fodci.stage1_baseline", "schema_version": "1.0", "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(), "model": identity, "dataset": {"path": str(args.dataset), "sha256": dataset_fingerprint, "records": len(records), "split": "benchmark"}, "protocol": protocol, "evaluation": evaluation}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "records": len(records), "parameters": parameter_count, **evaluation["aggregate"], "report": str(args.report), "markdown": str(args.markdown)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
