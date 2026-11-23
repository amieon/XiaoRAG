"""企业文档解析、DocumentIR 与知识单元构建。"""

from .document_ir import Block, DocumentIR, Page, Relation, SourceRef
from .knowledge_units import (
    ElementLocation,
    KnowledgeUnitBuildResult,
    KnowledgeUnitConfig,
    KnowledgeUnit,
    build_knowledge_units,
    build_knowledge_units_from_json,
    validate_knowledge_units,
    write_knowledge_units,
)
from .pipeline import PageParseResult, ParserConfig, RegionPipeline

__all__ = [
    "Block", "DocumentIR", "Page", "Relation", "SourceRef",
    "PageParseResult", "ParserConfig", "RegionPipeline",
    "ElementLocation", "KnowledgeUnitBuildResult", "KnowledgeUnitConfig", "KnowledgeUnit",
    "build_knowledge_units", "build_knowledge_units_from_json", "validate_knowledge_units", "write_knowledge_units",
]
