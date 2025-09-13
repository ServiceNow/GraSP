"""Production-ready pipeline for normalizing, clustering, and merging instruction tags.

Key features
------------
- Clean, testable architecture with dataclasses and pure functions
- Deterministic canonical selection within clusters (most frequent > shortest > lexicographic)
- Optional GPU DBSCAN via RAPIDS cuML (falls back to scikit-learn)
- Optional Hugging Face auth via HF_TOKEN env var (no hard dependency)
- Robust handling of edge cases (empty inputs, sparse rules)
- Clear stats output + metadata updates (boolean flags)
- CLI for batch processing JSON files or STDIN

Example
-------
python filter_tags.py --input sample.json --output stats.json

Input JSON format (list of records):
[
  {"id": 1, "metadata": {"data_taxonomy": {"instruction_tags": ["Tokenization", "Tokenizer", "BERT Models"]}}},
  {"id": 2, "metadata": {"data_taxonomy": {"instruction_tags": ["bert-model", "embedding", "Tokenize"]}}},
  {"id": 3, "metadata": {"data_taxonomy": {"instruction_tags": ["translation", "machine translation", "nlp"]}}}
]

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
from mlxtend.frequent_patterns import association_rules, fpgrowth  # type: ignore[import-untyped]
from mlxtend.preprocessing import TransactionEncoder  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

# ----------------------------
# Logging
# ----------------------------
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("instag_pipeline")


# ----------------------------
# Utility
# ----------------------------


def convert_seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string (utility)."""
    import time as _time

    return _time.strftime("%H:%M:%S", _time.gmtime(seconds))


# ----------------------------
# Configuration
# ----------------------------


@dataclass(frozen=True)
class ClusterConfig:
    model_name: str = "whaleloops/phrase-bert"
    eps: float = 0.05
    min_samples: int = 2
    use_gpu: bool = False


@dataclass(frozen=True)
class AssocConfig:
    min_support: float = 0.4
    min_confidence: float = 0.99


@dataclass(frozen=True)
class PipelineConfig:
    frequency_alpha: int = 1
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    assoc: AssocConfig = field(default_factory=AssocConfig)


# ----------------------------
# Preprocessing
# ----------------------------


@dataclass
class PreprocessResult:
    # Per-record normalized tags
    normalized_by_record: List[List[str]]
    # Mapping from normalized tag -> Counter of original variants
    norm_to_originals: Dict[str, Counter]
    # Unique normalized tags (sorted for determinism)
    unique_normalized: List[str]


class TagPreprocessor:
    """Lowercase, strip non-alphanumerics, stem; track original variants per normalized form."""

    def __init__(self) -> None:
        self._stemmer = PorterStemmer()
        self._pattern = re.compile(r"[^a-zA-Z0-9]+")

    def preprocess_one(self, tag: str) -> str:
        norm = self._pattern.sub(" ", tag.lower()).strip()
        stemmed = " ".join(self._stemmer.stem(tok) for tok in norm.split())
        return stemmed

    def run(self, records: Sequence[Sequence[str]]) -> PreprocessResult:
        norm_to_originals: Dict[str, Counter] = defaultdict(Counter)
        normalized_by_record: List[List[str]] = []

        for rec in records:
            norm_rec: List[str] = []
            for tag in rec:
                n = self.preprocess_one(tag)
                norm_rec.append(n)
                norm_to_originals[n][tag] += 1
            normalized_by_record.append(norm_rec)

        unique_norm = sorted({t for rec in normalized_by_record for t in rec})
        return PreprocessResult(
            normalized_by_record=normalized_by_record,
            norm_to_originals=norm_to_originals,
            unique_normalized=unique_norm,
        )


# ----------------------------
# Embeddings + Clustering
# ----------------------------


class SemanticClusterer:
    def __init__(self, cfg: ClusterConfig) -> None:
        self.cfg = cfg
        self._device = "cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu"
        token = os.getenv("HF_TOKEN")
        try:
            # Attempt HF auth only if token present; avoid hard dependency
            if token:
                try:
                    from huggingface_hub import (  # local import to avoid hard dep at import time
                        login,
                    )

                    login(token=token)
                    logger.info("Authenticated to Hugging Face hub via HF_TOKEN.")
                except Exception as e:
                    logger.warning("HF login failed; continuing anonymously: %s", e)
            self._model = SentenceTransformer(self.cfg.model_name, device=self._device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SentenceTransformer '{self.cfg.model_name}': {e}"
            ) from e

    def _cluster_sklearn(self, X) -> List[int]:
        db = DBSCAN(eps=self.cfg.eps, min_samples=self.cfg.min_samples, metric="cosine")
        return cast(List[int], db.fit_predict(X).tolist())

    def _cluster_cuml(self, X) -> List[int]:
        try:
            import cupy as cp  # type: ignore[import-not-found]
            from cuml.cluster import DBSCAN as CuDBSCAN  # type: ignore[import-not-found]
        except Exception as e:
            logger.warning("cuML not available; falling back to scikit-learn DBSCAN: %s", e)
            return self._cluster_sklearn(X)

        db = CuDBSCAN(eps=self.cfg.eps, min_samples=self.cfg.min_samples, metric="cosine")
        labels = db.fit_predict(cp.asarray(X))
        return cast(List[int], cp.asnumpy(labels).astype(int).tolist())

    def fit_predict(self, tags: Sequence[str]) -> Dict[int, List[str]]:
        if not tags:
            return {}
        logger.info("Embedding %d tags on %s", len(tags), self._device)
        emb = self._model.encode(list(tags), convert_to_numpy=True, batch_size=128)
        if self._device == "cuda":
            labels = self._cluster_cuml(emb)
        else:
            labels = self._cluster_sklearn(emb)
        clusters: Dict[int, List[str]] = defaultdict(list)
        for t, lab in zip(tags, labels):
            clusters[int(lab)].append(t)
        # Deterministic ordering for reproducibility
        for lab in clusters:
            clusters[lab].sort()
        return dict(clusters)


# ----------------------------
# Association (FP-Growth + Rules)
# ----------------------------


@dataclass
class AssocResult:
    # Mapping of merged group token -> set of member tags
    groups: Dict[str, Set[str]]
    # Mapping of individual tag -> merged group token
    tag_to_group: Dict[str, str]


class AssociationMerger:
    def __init__(self, cfg: AssocConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _merge_overlapping(sets: List[Set[str]]) -> List[Set[str]]:
        """Merge overlapping sets until fixed point (union-find-lite)."""
        changed = True
        res = [set(s) for s in sets if s]
        while changed:
            changed = False
            out: List[Set[str]] = []
            for s in res:
                merged = False
                for o in out:
                    if s & o:
                        o |= s
                        merged = True
                        changed = True
                        break
                if not merged:
                    out.append(s)
            res = out
        return res

    def run(self, transactions: Sequence[Sequence[str]]) -> AssocResult:
        if not transactions:
            return AssocResult(groups={}, tag_to_group={})

        te = TransactionEncoder()
        te_arr = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_arr, columns=te.columns_).astype(bool)

        freq = fpgrowth(df, min_support=self.cfg.min_support, use_colnames=True)
        if freq.empty:
            return AssocResult(groups={}, tag_to_group={})

        rules = association_rules(freq, metric="confidence", min_threshold=self.cfg.min_confidence)
        if rules.empty:
            return AssocResult(groups={}, tag_to_group={})

        # Build candidate groups from rules, then merge overlaps
        raw_groups: List[Set[str]] = []
        for _, r in rules.iterrows():
            antecedent = set(r["antecedents"])  # type: ignore[assignment]
            consequent = set(r["consequents"])  # type: ignore[assignment]
            raw_groups.append(set(sorted(antecedent | consequent)))

        merged_groups = self._merge_overlapping(raw_groups)
        groups: Dict[str, Set[str]] = {}
        tag_to_group: Dict[str, str] = {}
        for g in merged_groups:
            token = " + ".join(sorted(g))
            groups[token] = set(sorted(g))
            for t in g:
                tag_to_group[t] = token
        return AssocResult(groups=groups, tag_to_group=tag_to_group)


# ----------------------------
# Canonical selection & application
# ----------------------------


def select_canonical_original(
    candidates: Iterable[str],
    norm_to_originals: Mapping[str, Counter],
    global_original_counts: Mapping[str, int],
) -> str:
    """Choose a human-friendly canonical variant for a set of normalized tags.

    Priority: highest global frequency > shortest string length > lexicographic.
    """
    # Collect all original variants for all normalized candidates
    originals: Counter = Counter()
    for n in candidates:
        originals.update(norm_to_originals.get(n, {}))

    if not originals:
        # Fallback: use first normalized as-is
        c = next(iter(candidates))
        return c

    # Compute global weight per original (sum per dataset + global_original_counts)
    scored: List[Tuple[str, Tuple[int, int, str]]] = []
    for orig, cnt in originals.items():
        total = cnt + int(global_original_counts.get(orig, 0))
        scored.append((orig, (total, -len(orig), orig)))

    # Sort by (-total, +len, lex) via reverse on first component using tuple tweak
    # We made length negative to prefer shorter when using max()
    best = max(scored, key=lambda x: x[1])
    return best[0]


# ----------------------------
# Stats helpers
# ----------------------------


def unique_tags_and_count(records: Sequence[Sequence[str]]) -> Tuple[int, List[str]]:
    s = sorted({t for rec in records for t in rec})
    return len(s), s


def average_tags_per_record(records: Sequence[Sequence[str]]) -> float:
    if not records:
        return 0.0
    return round(sum(len(rec) for rec in records) / len(records), 2)


# ----------------------------
# Core pipeline
# ----------------------------


@dataclass
class PipelineOutput:
    instag_stats: Dict[str, Any]
    # Optionally, mutated records could be returned too; here we focus on stats


class InstaGPipeline:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.clusterer = SemanticClusterer(cfg.cluster)
        self.merger = AssociationMerger(cfg.assoc)
        self.preproc = TagPreprocessor()

    @staticmethod
    def _extract_instruction_tags(output_data: List[Dict[str, Any]]) -> List[List[str]]:
        out: List[List[str]] = []
        for rec in output_data:
            md = rec.get("metadata", {})
            tax = md.get("data_taxonomy", {}) if isinstance(md, dict) else {}
            tags = tax.get("instruction_tags", []) if isinstance(tax, dict) else []
            if not isinstance(tags, list):
                tags = []
            out.append([t for t in tags if isinstance(t, str) and t.strip()])
        return out

    @staticmethod
    def _update_metadata(
        output_data: List[Dict[str, Any]],
        processed: List[List[str]],
        changed_flags: List[bool],
        update_original: bool = True,
    ) -> None:
        for rec, tags, changed in zip(output_data, processed, changed_flags):
            md = rec.setdefault("metadata", {})
            if not isinstance(md, dict):
                continue
            tax = md.setdefault("data_taxonomy", {})
            if isinstance(tax, dict):
                if update_original:
                    tax["instruction_tags"] = list(tags)
                else:
                    tax["instruction_tags_processed"] = list(tags)
                tax["are_tags_normalized"] = bool(changed)

    def run(self, output_data: List[Dict[str, Any]]) -> PipelineOutput:
        # Extract raw tags
        tag_lists = self._extract_instruction_tags(output_data)
        raw_unique_count, _ = unique_tags_and_count(tag_lists)

        # Frequency filter (on originals)
        flat_raw = [t for rec in tag_lists for t in rec]
        raw_counts = Counter(flat_raw)
        kept_set = {t for t, c in raw_counts.items() if c >= self.cfg.frequency_alpha}
        filtered_lists = [[t for t in rec if t in kept_set] for rec in tag_lists]

        # If all filtered out, short-circuit with empty stats
        if not any(filtered_lists):
            changed_flags = [len(orig) != 0 for orig in tag_lists]
            self._update_metadata(output_data, [[] for _ in tag_lists], changed_flags)
            stats = {
                "num_records": len(output_data),
                "num_unique_tags_before_normalizing": raw_unique_count,
                "num_unique_tags": 0,
                "avg_tags_per_record": 0.0,
                "unique_tags": [],
                "clustered_tags": {},
                "merged_tags": {},
            }
            return PipelineOutput(instag_stats=stats)

        # Preprocess
        pre = self.preproc.run(filtered_lists)

        # Clustering (on unique normalized)
        clusters = self.clusterer.fit_predict(pre.unique_normalized)

        # Build mapping normalized -> canonical original using deterministic selection
        global_original_counts = raw_counts  # use original totals
        norm_to_canonical: Dict[str, str] = {}

        # For each cluster, choose canonical among all members
        for label, members in clusters.items():
            if label == -1 or len(members) == 1:
                # noise or singleton: pick best variant of its own
                for n in members:
                    canon = select_canonical_original(
                        [n], pre.norm_to_originals, global_original_counts
                    )
                    norm_to_canonical[n] = canon
            else:
                canon = select_canonical_original(
                    members, pre.norm_to_originals, global_original_counts
                )
                for n in members:
                    norm_to_canonical[n] = canon

        # Clustered display map: canonical -> member originals (>=2 only)
        clustered_display: Dict[str, List[str]] = {}
        for label, members in clusters.items():
            if len(members) <= 1:
                continue
            canonical = select_canonical_original(
                members, pre.norm_to_originals, global_original_counts
            )
            member_originals: Set[str] = set()
            for n in members:
                member_originals.update(pre.norm_to_originals.get(n, {}).keys())
            clustered_display[canonical] = sorted(member_originals)

        # Apply normalization to records (map each normalized tag -> canonical original), dedupe deterministically
        normalized_lists: List[List[str]] = []
        for rec_norm in pre.normalized_by_record:
            mapped = [norm_to_canonical.get(n, n) for n in rec_norm]
            # Deterministic dedupe while preserving sorted order
            dedup = sorted(set(mapped))
            normalized_lists.append(dedup)

        # Association merging on normalized tags
        assoc_res = self.merger.run(normalized_lists)
        if assoc_res.groups:
            replaced_lists: List[List[str]] = []
            for rec in normalized_lists:
                repl = [assoc_res.tag_to_group.get(t, t) for t in rec]
                replaced_lists.append(sorted(set(repl)))
            final_lists = replaced_lists
        else:
            final_lists = normalized_lists

        # Update metadata
        changed_flags = [set(orig) != set(proc) for orig, proc in zip(tag_lists, final_lists)]
        self._update_metadata(output_data, final_lists, changed_flags, update_original=True)

        # Stats
        num_unique_after, unique_after = unique_tags_and_count(final_lists)
        avg_tags = average_tags_per_record(final_lists)

        stats = {
            "num_records": len(output_data),
            "num_unique_tags_before_normalizing": raw_unique_count,
            "num_unique_tags": num_unique_after,
            "avg_tags_per_record": avg_tags,
            "unique_tags": unique_after,
            "clustered_tags": clustered_display,
            "merged_tags": {k: sorted(v) for k, v in assoc_res.groups.items()},
        }

        return PipelineOutput(instag_stats=stats)


# ----------------------------
# Public API
# ----------------------------


def extract_instag_stats(
    output_data: List[Dict[str, Any]],
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper that runs the pipeline and returns stats as dict."""
    cfg = config or PipelineConfig()
    pipeline = InstaGPipeline(cfg)
    out = pipeline.run(output_data)
    return {"instag_stats": out.instag_stats}


# ----------------------------
# CLI
# ----------------------------


def _read_input(
    path: Union[int, Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]],
) -> List[Dict[str, Any]]:
    if path in (None, "-"):
        raw = sys.stdin.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records")
    return data  # type: ignore[return-value]


def _write_output(obj: list[dict[str, Any]], path: Optional[str]) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    if path in (None, "-"):
        sys.stdout.write(text + "\n")
    else:
        out_path = cast(str, path)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize & analyze instruction tags")
    p.add_argument(
        "--input",
        "-i",
        help="Path to input JSON (list of records); '-' for STDIN",
        default=None,
    )
    p.add_argument(
        "--output",
        "-o",
        help="Path to output JSON with updated tags; omit to update input file",
        default=None,
    )

    p.add_argument(
        "--model",
        default=ClusterConfig.model_name,
        help="SentenceTransformer model name",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=ClusterConfig.eps,
        help="DBSCAN eps (cosine distance)",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=ClusterConfig.min_samples,
        help="DBSCAN min_samples",
    )
    p.add_argument("--gpu", action="store_true", help="Use GPU (cuML DBSCAN if available)")

    p.add_argument(
        "--min-support",
        type=float,
        default=AssocConfig.min_support,
        help="FP-Growth min_support",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=AssocConfig.min_confidence,
        help="Association rules min_confidence",
    )

    p.add_argument(
        "--alpha",
        type=int,
        default=PipelineConfig.frequency_alpha,
        help="Frequency filter alpha",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    cfg = PipelineConfig(
        frequency_alpha=args.alpha,
        cluster=ClusterConfig(
            model_name=args.model,
            eps=args.eps,
            min_samples=args.min_samples,
            use_gpu=args.gpu,
        ),
        assoc=AssocConfig(
            min_support=args.min_support,
            min_confidence=args.min_confidence,
        ),
    )

    try:
        data = _read_input(args.input)
        stats = extract_instag_stats(data, cfg)
        logger.info(json.dumps(stats, indent=2))

        # Write the updated data back to the input file or output file
        if args.output:
            _write_output(data, args.output)
            logger.info(f"Updated data written to {args.output}")
        else:
            # If no output file specified, update the input file directly
            _write_output(data, args.input)
            logger.info(f"Updated data written back to {args.input}")

        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    raise sys.exit(main())
