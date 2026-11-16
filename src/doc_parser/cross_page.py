"""Initial rule-based recovery of relationships across document pages.

The module intentionally produces auditable relations instead of destructively
merging blocks. It is a first-pass document assembler; ambiguous candidates
remain in the IR with a lower confidence and a ``candidate`` decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any

from .document_ir import Block, DocumentIR, Relation


TEXT_BLOCK_TYPES = {"paragraph", "list"}
SKIP_TYPES = {
    "document_title",
    "section_heading",
    "header",
    "footer",
    "page_number",
    "figure",
    "figure_caption",
    "formula",
    "table_caption",
}
_HEADING_OR_LIST = re.compile(r"^(?:#{1,6}\s|\d+(?:\.\d+)*[.)、．]\s*|[（(]?[一二三四五六七八九十百千万]+[）)、．.]\s*)")
_STRONG_END = re.compile(r"[。！？!?]\s*$")
_WEAK_END = re.compile(r"[,，:：;；、\-—]\s*$")


@dataclass(slots=True)
class RecoveryStats:
    """Counters and auditable examples from one recovery run."""

    pages: int = 0
    adjacent_page_pairs: int = 0
    considered_text_pairs: int = 0
    considered_table_pairs: int = 0
    relation_counts: dict[str, int] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def record(self, relation: Relation, *, page_pair: tuple[int, int]) -> None:
        self.relation_counts[relation.relation_type] = self.relation_counts.get(relation.relation_type, 0) + 1
        if len(self.examples) < 100:
            self.examples.append({
                "relation_id": relation.relation_id,
                "relation_type": relation.relation_type,
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "confidence": relation.confidence,
                "decision": relation.metadata.get("decision"),
                "page_pair": list(page_pair),
                "evidence": relation.evidence,
            })


def _horizontal_overlap(left: Block, right: Block) -> float:
    left_width = max(1.0, left.bbox[2] - left.bbox[0])
    right_width = max(1.0, right.bbox[2] - right.bbox[0])
    intersection = max(0.0, min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]))
    return intersection / min(left_width, right_width)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _eligible_text(block: Block) -> bool:
    return bool(block.text.strip()) and block.block_type in TEXT_BLOCK_TYPES


def _boundary_candidates(page_blocks: list[Block], *, height: float, top: bool) -> list[Block]:
    eligible = [block for block in page_blocks if _eligible_text(block)]
    if top:
        result = [block for block in eligible if block.bbox[1] <= height * 0.42]
        return sorted(result or eligible, key=lambda item: (item.bbox[1], item.reading_order or 10**9))[:4]
    result = [block for block in eligible if block.bbox[3] >= height * 0.58]
    return sorted(result or eligible, key=lambda item: (item.bbox[3], item.reading_order or 10**9), reverse=True)[:4]


def _text_continuation_score(previous: Block, following: Block, previous_height: float, following_height: float) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    previous_bottom = previous.bbox[3] / max(previous_height, 1.0)
    following_top = following.bbox[1] / max(following_height, 1.0)
    if previous_bottom >= 0.70:
        score += 0.22
        evidence.append("previous block is near page bottom")
    elif previous_bottom >= 0.58:
        score += 0.12
        evidence.append("previous block is in lower page region")
    if following_top <= 0.30:
        score += 0.22
        evidence.append("following block is near page top")
    elif following_top <= 0.42:
        score += 0.12
        evidence.append("following block is in upper page region")
    overlap = _horizontal_overlap(previous, following)
    if overlap >= 0.75:
        score += 0.22
        evidence.append("horizontal spans align")
    elif overlap >= 0.45:
        score += 0.12
        evidence.append("horizontal spans partly align")
    if previous.block_type == following.block_type:
        score += 0.08
        evidence.append("block types match")
    previous_text = previous.text.strip()
    following_text = following.text.strip()
    if _WEAK_END.search(previous_text) or not re.search(r"[.!?。！？]$", previous_text):
        score += 0.16
        evidence.append("previous text looks unfinished")
    elif _STRONG_END.search(previous_text):
        evidence.append("previous text has sentence-ending punctuation")
    if following_text and not _HEADING_OR_LIST.match(following_text):
        score += 0.10
        evidence.append("following text is not a new heading or list")
    return min(score, 1.0), evidence


def _table_continuation_score(previous: Block, following: Block, previous_height: float, following_height: float) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    if previous.bbox[3] / max(previous_height, 1.0) >= 0.55:
        score += 0.25
        evidence.append("previous table reaches lower page region")
    if following.bbox[1] / max(following_height, 1.0) <= 0.45:
        score += 0.20
        evidence.append("following table starts in upper page region")
    overlap = _horizontal_overlap(previous, following)
    if overlap >= 0.75:
        score += 0.35
        evidence.append("table horizontal spans align")
    elif overlap >= 0.45:
        score += 0.18
        evidence.append("table horizontal spans partly align")
    if previous.text and following.text:
        score += 0.10
        evidence.append("both table blocks contain content")
    return min(score, 1.0), evidence


def _make_relation(relation_number: int, relation_type: str, source: Block, target: Block, score: float, evidence: list[str]) -> Relation:
    decision = "accept" if score >= 0.75 else "candidate"
    return Relation(
        relation_id=f"rel_cross_{relation_number:06d}",
        relation_type=relation_type,
        source_id=source.block_id,
        target_id=target.block_id,
        confidence=round(score, 4),
        evidence=evidence,
        metadata={"decision": decision, "method": "rule_v0.1"},
    )


def recover_cross_page_relations(document: DocumentIR) -> tuple[DocumentIR, RecoveryStats]:
    """Return a copy of ``document`` with first-pass cross-page relations.

    Only adjacent, consecutive page numbers are considered. This prevents a
    sparse benchmark sample (for example pages 10 and 55 from one book) from
    being treated as if it were a true page break.
    """

    recovered = deepcopy(document)
    recovered.validate()
    pages = sorted(recovered.pages, key=lambda page: page.page_no)
    stats = RecoveryStats(pages=len(pages))
    relation_number = 1
    for previous_page, following_page in zip(pages, pages[1:]):
        if following_page.page_no != previous_page.page_no + 1:
            continue
        stats.adjacent_page_pairs += 1
        previous_blocks = sorted(previous_page.blocks, key=lambda block: block.reading_order if block.reading_order is not None else 10**9)
        following_blocks = sorted(following_page.blocks, key=lambda block: block.reading_order if block.reading_order is not None else 10**9)

        # Repeated headers and footers are document-level noise relations.
        for previous in previous_blocks:
            if not previous.text.strip() or previous.block_type not in {"paragraph", "section_heading"}:
                continue
            for following in following_blocks:
                if _normalise_text(previous.text) != _normalise_text(following.text):
                    continue
                previous_top = previous.bbox[1] / max(previous_page.height, 1.0)
                following_top = following.bbox[1] / max(following_page.height, 1.0)
                previous_bottom = previous.bbox[3] / max(previous_page.height, 1.0)
                following_bottom = following.bbox[3] / max(following_page.height, 1.0)
                if (previous_top < 0.18 and following_top < 0.18) or (previous_bottom > 0.82 and following_bottom > 0.82):
                    relation = _make_relation(relation_number, "duplicate_header_footer", previous, following, 0.95, ["normalized text repeats at the same page edge"])
                    relation_number += 1
                    recovered.relations.append(relation)
                    stats.record(relation, page_pair=(previous_page.page_no, following_page.page_no))

        previous_text_blocks = _boundary_candidates(previous_blocks, height=previous_page.height, top=False)
        following_text_blocks = _boundary_candidates(following_blocks, height=following_page.height, top=True)
        best_text: tuple[float, Block, Block, list[str]] | None = None
        for previous in previous_text_blocks:
            for following in following_text_blocks:
                stats.considered_text_pairs += 1
                score, evidence = _text_continuation_score(previous, following, previous_page.height, following_page.height)
                if best_text is None or score > best_text[0]:
                    best_text = (score, previous, following, evidence)
        if best_text and best_text[0] >= 0.50:
            score, previous, following, evidence = best_text
            relation = _make_relation(relation_number, "paragraph_continuation", previous, following, score, evidence)
            relation_number += 1
            recovered.relations.append(relation)
            stats.record(relation, page_pair=(previous_page.page_no, following_page.page_no))

        previous_tables = [block for block in previous_blocks if block.block_type == "table"]
        following_tables = [block for block in following_blocks if block.block_type == "table"]
        best_table: tuple[float, Block, Block, list[str]] | None = None
        for previous in previous_tables:
            for following in following_tables:
                stats.considered_table_pairs += 1
                score, evidence = _table_continuation_score(previous, following, previous_page.height, following_page.height)
                if best_table is None or score > best_table[0]:
                    best_table = (score, previous, following, evidence)
        if best_table and best_table[0] >= 0.55:
            score, previous, following, evidence = best_table
            relation = _make_relation(relation_number, "cross_page_table", previous, following, score, evidence)
            relation_number += 1
            recovered.relations.append(relation)
            stats.record(relation, page_pair=(previous_page.page_no, following_page.page_no))

    recovered.validate()
    return recovered, stats
