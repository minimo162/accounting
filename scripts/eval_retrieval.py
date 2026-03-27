"""Evaluate retrieval quality or bootstrap annotation candidates."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.arag.config import Config
from src.arag.query_rewrite import QueryExpander
from src.arag.reranker import BaseReranker, HeuristicReranker, LLMReranker
from src.arag.retrieval import ChunkCorpus
from src.arag.tools.hybrid_search import HybridSearchTool
from src.arag.tools.keyword_search import KeywordSearchTool
from src.arag.tools.semantic_search import SemanticSearchTool
from src.arag.llm import LLMClient
from src.embedding import create_embedder


def recall_at_k(results: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold.intersection(results[:k]) else 0.0


def reciprocal_rank(results: list[str], gold: set[str]) -> float:
    for idx, result in enumerate(results, start=1):
        if result in gold:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(results: list[str], gold: set[str], k: int) -> float:
    dcg = 0.0
    for idx, result in enumerate(results[:k], start=1):
        if result in gold:
            dcg += 1.0 / math.log2(idx + 1)
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


def load_eval_queries(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="data/eval_queries.jsonl")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--index", default="data/index/sentence_index.pkl")
    parser.add_argument("--mode", choices=["semantic", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--write-candidates", action="store_true")
    args = parser.parse_args()

    config = Config.from_env()
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    corpus = ChunkCorpus(chunks)
    keyword_tool = KeywordSearchTool(corpus)
    semantic_tool = None
    hybrid_tool = None
    if args.mode in {"semantic", "hybrid"}:
        embedder = create_embedder(config.embedding)
        semantic_tool = SemanticSearchTool(args.index, embedder.embed_query, corpus)
    if args.mode == "hybrid":
        llm = LLMClient(config.llm) if config.retrieval.reranker == "llm" or config.retrieval.enable_hyde else None
        if config.retrieval.reranker == "llm" and llm:
            reranker = LLMReranker(llm)
        elif config.retrieval.reranker == "none":
            reranker = BaseReranker()
        else:
            reranker = HeuristicReranker()
        hybrid_tool = HybridSearchTool(
            semantic_tool=semantic_tool,
            keyword_tool=keyword_tool,
            query_expander=QueryExpander(config.retrieval, llm=llm),
            reranker=reranker,
            config=config.retrieval,
        )

    queries = load_eval_queries(Path(args.queries))
    scored = []
    bootstrap_rows = []

    for entry in queries:
        query = entry["query"]
        gold = set(entry.get("gold_parent_ids", []) or entry.get("gold_chunk_ids", []))
        if args.mode == "semantic":
            assert semantic_tool is not None
            results = semantic_tool.search(query, args.top_k)
            result_ids = [result.parent_id for result in results]
        elif args.mode == "keyword":
            results = keyword_tool.search(entry.get("keywords") or [query], args.top_k)
            result_ids = [result.parent_id for result in results]
        else:
            assert hybrid_tool is not None
            results, _, _ = hybrid_tool.search(query, args.top_k)
            result_ids = [result.parent_id for result in results]

        if args.write_candidates:
            bootstrap_rows.append(
                {
                    **entry,
                    "candidate_parent_ids": result_ids,
                }
            )

        if gold:
            scored.append(
                {
                    "query": query,
                    "recall@5": recall_at_k(result_ids, gold, 5),
                    "recall@10": recall_at_k(result_ids, gold, 10),
                    "mrr": reciprocal_rank(result_ids, gold),
                    "ndcg@10": ndcg_at_k(result_ids, gold, 10),
                }
            )

    if args.write_candidates:
        out_path = Path(args.queries).with_suffix(".candidates.json")
        out_path.write_text(json.dumps(bootstrap_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote candidate file: {out_path}")

    if not scored:
        print("No queries with gold labels found. Use --write-candidates to bootstrap annotation.")
        return

    summary = {
        metric: sum(row[metric] for row in scored) / len(scored)
        for metric in ("recall@5", "recall@10", "mrr", "ndcg@10")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
