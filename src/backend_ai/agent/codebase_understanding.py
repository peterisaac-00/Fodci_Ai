"""Phase 12.2 bounded, task-aware codebase understanding.

This module interprets evidence returned by existing read-only tools. It does not
execute project code, modify files, build a long-context store, or replace the
planner. Analysis is deterministic, UTF-8 safe, and intentionally incomplete
when repository evidence is insufficient.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import re
from pathlib import Path
from typing import Any

from backend_ai.tools.base import ToolError
from backend_ai.tools.project_context import ProjectContext, ProjectContextBuilder
from backend_ai.tools.project_structure import ProjectStructureResult, project_structure
from backend_ai.tools.read_file import ReadFileResult, read_file
from backend_ai.tools.search_code import SearchCodeResult, search_code


DEFAULT_MAX_FILES = 512
DEFAULT_MAX_INSPECTED_FILES = 48
DEFAULT_MAX_FILE_BYTES = 131_072
DEFAULT_MAX_SYMBOLS = 256
DEFAULT_MAX_REFERENCES = 512
DEFAULT_MAX_DEPENDENCIES = 256
DEFAULT_MAX_RELEVANT_FILES = 32
DEFAULT_MAX_EVIDENCE = 256
MAX_MAX_FILES = 4_096
MAX_MAX_INSPECTED_FILES = 256
MAX_MAX_FILE_BYTES = 1_048_576
MAX_MAX_SYMBOLS = 2_048
MAX_MAX_REFERENCES = 4_096
MAX_MAX_DEPENDENCIES = 2_048
MAX_MAX_RELEVANT_FILES = 256
MAX_MAX_EVIDENCE = 2_048


class UnderstandingConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class UnderstandingCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class UnderstandingEvidence:
    """A bounded repository fact supporting one or more understanding claims."""

    kind: str
    path: str
    detail: str
    line_start: int | None = None
    line_end: int | None = None
    confidence: UnderstandingConfidence = UnderstandingConfidence.MEDIUM

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.path.strip() or not self.detail.strip():
            raise ValueError("evidence kind, path, and detail must contain text")
        if self.line_start is not None and self.line_start < 1:
            raise ValueError("line_start must be positive when supplied")
        if self.line_end is not None and self.line_end < 1:
            raise ValueError("line_end must be positive when supplied")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "detail": self.detail,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "confidence": self.confidence.value,
        }


@dataclass(frozen=True, slots=True)
class RelevantFile:
    path: str
    role: str
    relevance: str
    reasons: tuple[str, ...]
    evidence: tuple[UnderstandingEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.relevance not in {"high", "medium", "low"}:
            raise ValueError("relevance must be high, medium, or low")
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(self.reasons)))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "relevance": self.relevance,
            "reasons": list(self.reasons),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class SymbolInfo:
    name: str
    kind: str
    path: str
    line_start: int
    line_end: int
    signature: str
    confidence: UnderstandingConfidence
    evidence: tuple[UnderstandingEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ReferenceInfo:
    source_path: str
    target: str
    relation: str
    line: int | None
    confidence: UnderstandingConfidence
    evidence: tuple[UnderstandingEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target": self.target,
            "relation": self.relation,
            "line": self.line,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class DependencyInfo:
    source: str
    target: str
    kind: str
    confidence: UnderstandingConfidence
    evidence: tuple[UnderstandingEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class ArchitectureLayer:
    name: str
    paths: tuple[str, ...]
    relationships: tuple[str, ...]
    confidence: UnderstandingConfidence
    evidence: tuple[UnderstandingEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paths": list(self.paths),
            "relationships": list(self.relationships),
            "confidence": self.confidence.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True, slots=True)
class CodebaseUnderstanding:
    """Bounded structural knowledge for one explicit repository and task."""

    root: Path
    task: str
    project_type: str
    frameworks: tuple[str, ...]
    entry_points: tuple[str, ...]
    important_directories: tuple[str, ...]
    important_files: tuple[str, ...]
    components: tuple[str, ...]
    symbols: tuple[SymbolInfo, ...]
    references: tuple[ReferenceInfo, ...]
    dependencies: tuple[DependencyInfo, ...]
    architecture: tuple[ArchitectureLayer, ...]
    relevant_files: tuple[RelevantFile, ...]
    evidence: tuple[UnderstandingEvidence, ...]
    confidence: UnderstandingConfidence
    completeness: UnderstandingCompleteness
    warnings: tuple[str, ...] = ()
    truncated: bool = False
    truncation_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frameworks", _unique_text(self.frameworks))
        object.__setattr__(self, "entry_points", _unique_text(self.entry_points))
        object.__setattr__(self, "important_directories", _unique_text(self.important_directories))
        object.__setattr__(self, "important_files", _unique_text(self.important_files))
        object.__setattr__(self, "components", _unique_text(self.components))
        object.__setattr__(self, "warnings", _unique_text(self.warnings))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "architecture", tuple(self.architecture))
        object.__setattr__(self, "relevant_files", tuple(self.relevant_files))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "task": self.task,
            "project_type": self.project_type,
            "frameworks": list(self.frameworks),
            "entry_points": list(self.entry_points),
            "important_directories": list(self.important_directories),
            "important_files": list(self.important_files),
            "components": list(self.components),
            "symbols": [item.to_dict() for item in self.symbols],
            "references": [item.to_dict() for item in self.references],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "architecture": [item.to_dict() for item in self.architecture],
            "relevant_files": [item.to_dict() for item in self.relevant_files],
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence.value,
            "completeness": self.completeness.value,
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }

    def compact_summary(self) -> str:
        """Return bounded facts suitable for planner context, not raw file contents."""

        files = ", ".join(item.path for item in self.relevant_files[:8]) or "none confirmed"
        symbols = ", ".join(f"{item.name} ({item.path})" for item in self.symbols[:8]) or "none confirmed"
        layers = ", ".join(item.name for item in self.architecture[:8]) or "unknown"
        return (
            f"project_type={self.project_type}; frameworks={','.join(self.frameworks) or 'unknown'}; "
            f"entry_points={','.join(self.entry_points) or 'none confirmed'}; relevant_files={files}; "
            f"symbols={symbols}; architecture={layers}; confidence={self.confidence.value}; "
            f"completeness={self.completeness.value}"
        )


class CodebaseUnderstandingBuilder:
    """Build and update bounded repository understanding from existing read-only tools."""

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        max_inspected_files: int = DEFAULT_MAX_INSPECTED_FILES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        max_symbols: int = DEFAULT_MAX_SYMBOLS,
        max_references: int = DEFAULT_MAX_REFERENCES,
        max_dependencies: int = DEFAULT_MAX_DEPENDENCIES,
        max_relevant_files: int = DEFAULT_MAX_RELEVANT_FILES,
        max_evidence: int = DEFAULT_MAX_EVIDENCE,
    ) -> None:
        self.max_files = _limit("max_files", max_files, 1, MAX_MAX_FILES)
        self.max_inspected_files = _limit("max_inspected_files", max_inspected_files, 1, MAX_MAX_INSPECTED_FILES)
        self.max_file_bytes = _limit("max_file_bytes", max_file_bytes, 1, MAX_MAX_FILE_BYTES)
        self.max_symbols = _limit("max_symbols", max_symbols, 1, MAX_MAX_SYMBOLS)
        self.max_references = _limit("max_references", max_references, 1, MAX_MAX_REFERENCES)
        self.max_dependencies = _limit("max_dependencies", max_dependencies, 1, MAX_MAX_DEPENDENCIES)
        self.max_relevant_files = _limit("max_relevant_files", max_relevant_files, 1, MAX_MAX_RELEVANT_FILES)
        self.max_evidence = _limit("max_evidence", max_evidence, 1, MAX_MAX_EVIDENCE)

    def build(
        self,
        task: str,
        project_root: Path | str,
        *,
        project_context: ProjectContext | None = None,
    ) -> CodebaseUnderstanding:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must contain text")
        root = Path(project_root).expanduser().resolve(strict=False)
        context = project_context or ProjectContextBuilder().build(root)
        structure = project_structure(
            root,
            max_files=self.max_files,
            max_inspected_files=min(self.max_inspected_files, 64),
            max_file_bytes=self.max_file_bytes,
        )
        paths = tuple(sorted(structure.project_files, key=str.casefold))[: self.max_files]
        task_tokens = _task_tokens(task)
        relevant = self._rank_files(task_tokens, paths, structure)
        inspection_paths = _inspection_paths(relevant, structure, self.max_inspected_files)
        texts, read_warnings = _read_texts(root, inspection_paths, self.max_file_bytes)
        symbols = _analyze_symbols(texts, self.max_symbols)
        references = _analyze_references(texts, symbols, self.max_references)
        dependencies = _analyze_dependencies(texts, self.max_dependencies)
        evidence = _root_evidence(structure, context, relevant, symbols, references, dependencies, self.max_evidence)
        components = _components(structure, paths, symbols)
        architecture = _architecture(paths, symbols, dependencies, self.max_evidence)
        relevant = self._enrich_relevance(relevant, symbols, references, dependencies)
        warnings = list(structure.warnings) + list(context.warnings) + list(read_warnings)
        if len(paths) >= self.max_files or structure.truncated:
            warnings.append("bounded project discovery may omit files outside the selected understanding budget")
        completeness = UnderstandingCompleteness.PARTIAL if structure.truncated or read_warnings else UnderstandingCompleteness.COMPLETE
        confidence = _understanding_confidence(structure, context, evidence, symbols)
        return CodebaseUnderstanding(
            root=structure.root,
            task=_bounded(task, 4_000),
            project_type=structure.project_type or context.project_type or "unknown",
            frameworks=tuple(item.name for item in structure.frameworks),
            entry_points=tuple(item.name for item in structure.entry_points),
            important_directories=tuple(item.relative_path for item in structure.directories if item.category != "other")[:64],
            important_files=structure.important_files[:64],
            components=components,
            symbols=symbols,
            references=references,
            dependencies=dependencies,
            architecture=architecture,
            relevant_files=relevant,
            evidence=evidence,
            confidence=confidence,
            completeness=completeness,
            warnings=tuple(sorted(set(warnings))),
            truncated=structure.truncated or bool(read_warnings),
            truncation_reason=structure.truncation_reason or ("targeted_read_limit" if read_warnings else None),
        )

    def update_from_tool_result(self, understanding: CodebaseUnderstanding, tool_name: str, data: Any) -> CodebaseUnderstanding:
        """Merge newly observed read-only tool evidence without rescanning unrelated files."""

        if not isinstance(understanding, CodebaseUnderstanding):
            raise TypeError("understanding must be CodebaseUnderstanding")
        if isinstance(data, ProjectStructureResult):
            new_files = tuple(dict.fromkeys((*understanding.important_files, *data.important_files)))[:64]
            new_entry_points = tuple(dict.fromkeys((*understanding.entry_points, *(item.name for item in data.entry_points))))
            evidence = _bounded_evidence((*understanding.evidence, UnderstandingEvidence("tool_result", "project_structure", "updated structural project evidence", confidence=UnderstandingConfidence.HIGH)), self.max_evidence)
            return replace(understanding, project_type=data.project_type or understanding.project_type, frameworks=tuple(dict.fromkeys((*understanding.frameworks, *(item.name for item in data.frameworks)))), entry_points=new_entry_points, important_files=new_files, evidence=evidence, warnings=tuple(dict.fromkeys((*understanding.warnings, *data.warnings))))
        if isinstance(data, SearchCodeResult):
            additions = tuple(match.relative_path for match in data.matches)
            relevant = tuple(dict.fromkeys((*[item.path for item in understanding.relevant_files], *additions)))
            evidence = list(understanding.evidence)
            for match in data.matches[:16]:
                evidence.append(UnderstandingEvidence("search_match", match.relative_path, f"{data.query} matched repository text", match.line_number, match.line_number, UnderstandingConfidence.MEDIUM))
            return replace(understanding, relevant_files=_relevant_from_paths(relevant[: self.max_relevant_files]), evidence=_bounded_evidence(evidence, self.max_evidence), warnings=tuple(dict.fromkeys((*understanding.warnings, *data.skipped_reasons))))
        if isinstance(data, ReadFileResult):
            extra_symbols = _analyze_symbols({data.relative_path: data.content}, self.max_symbols)
            symbols = _merge_symbols(understanding.symbols, extra_symbols, self.max_symbols)
            evidence = (*understanding.evidence, UnderstandingEvidence("read_file", data.relative_path, "targeted file content was inspected", confidence=UnderstandingConfidence.HIGH))
            return replace(understanding, symbols=symbols, evidence=_bounded_evidence(evidence, self.max_evidence))
        return understanding

    def _rank_files(self, tokens: tuple[str, ...], paths: tuple[str, ...], structure: ProjectStructureResult) -> tuple[RelevantFile, ...]:
        entries: list[tuple[int, RelevantFile]] = []
        framework_names = {item.name.casefold() for item in structure.frameworks}
        entry_point_names = {item.name for item in structure.entry_points}
        for path in paths:
            lowered = path.casefold()
            name = Path(path).name.casefold()
            reasons: list[str] = []
            score = 0
            hits = [token for token in tokens if token in lowered]
            if hits:
                score += 8 + 2 * len(hits)
                reasons.append("task keyword matches path")
            role = _path_role(path)
            if role != "unknown":
                score += 2
                reasons.append(f"backend role: {role}")
            if path in structure.important_files or path in entry_point_names:
                score += 5
                reasons.append("structural project evidence")
            if _is_test_path(path) and any(token in {"test", "bug", "fix", "endpoint", "auth", "authentication", "database"} for token in tokens):
                score += 4
                reasons.append("task-relevant test path")
            if any(framework in lowered for framework in framework_names if framework not in {"python", "node.js"}):
                score += 1
                reasons.append("framework-related path")
            if score == 0:
                score = 1
                reasons.append("bounded structural candidate")
            relevance = "high" if score >= 10 else "medium" if score >= 5 else "low"
            evidence = (UnderstandingEvidence("path", path, "; ".join(reasons), confidence=UnderstandingConfidence.HIGH if score >= 10 else UnderstandingConfidence.MEDIUM),)
            entries.append((score, RelevantFile(path, role, relevance, tuple(reasons), evidence)))
        entries.sort(key=lambda item: (-item[0], item[1].path.casefold()))
        return tuple(item[1] for item in entries[: self.max_relevant_files])

    def _enrich_relevance(self, relevant: tuple[RelevantFile, ...], symbols: tuple[SymbolInfo, ...], references: tuple[ReferenceInfo, ...], dependencies: tuple[DependencyInfo, ...]) -> tuple[RelevantFile, ...]:
        symbol_paths = {item.path for item in symbols}
        reference_paths = {item.source_path for item in references}
        dependency_paths = {item.source for item in dependencies}
        result: list[RelevantFile] = []
        for item in relevant:
            reasons = list(item.reasons)
            score = 0
            if item.path in symbol_paths:
                reasons.append("contains detected symbols")
                score += 2
            if item.path in reference_paths or item.path in dependency_paths:
                reasons.append("participates in detected relationships")
                score += 2
            relevance = "high" if item.relevance == "high" or score >= 3 else item.relevance
            result.append(replace(item, relevance=relevance, reasons=tuple(dict.fromkeys(reasons))))
        return tuple(result)


# Public convenience API.
def understand_codebase(task: str, project_root: Path | str, *, project_context: ProjectContext | None = None, **limits: int) -> CodebaseUnderstanding:
    return CodebaseUnderstandingBuilder(**limits).build(task, project_root, project_context=project_context)


def _limit(name: str, value: int, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded(value: str, limit: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    return encoded[: max(0, limit - 3)].decode("utf-8", errors="replace") + "..."


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if isinstance(value, str) and value.strip()))


def _task_tokens(task: str) -> tuple[str, ...]:
    ignored = {"the", "and", "for", "with", "from", "into", "this", "that", "add", "fix", "make", "update", "change"}
    values = re.findall(r"[A-Za-z0-9_]{3,}", task.casefold())
    return tuple(dict.fromkeys(value for value in values if value not in ignored))[:24]


def _inspection_paths(relevant: Sequence[RelevantFile], structure: ProjectStructureResult, limit: int) -> tuple[str, ...]:
    candidates = list(item.path for item in relevant)
    candidates.extend(structure.important_files)
    candidates.extend(item.relative_path for item in structure.directories if item.category in {"source", "tests", "configuration"})
    return tuple(dict.fromkeys(path for path in candidates if Path(path).suffix.casefold() in {".py", ".pyw", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}))[:limit]


def _read_texts(root: Path, paths: Sequence[str], max_bytes: int) -> tuple[dict[str, str], tuple[str, ...]]:
    texts: dict[str, str] = {}
    warnings: list[str] = []
    for path in paths:
        try:
            result = read_file(root, path, max_bytes=max_bytes)
        except ToolError as exc:
            warnings.append(f"targeted read skipped {path}: {exc.code.value}")
            continue
        texts[path] = result.content
    return texts, tuple(warnings)


def _analyze_symbols(texts: Mapping[str, str], limit: int) -> tuple[SymbolInfo, ...]:
    symbols: list[SymbolInfo] = []
    for path in sorted(texts, key=str.casefold):
        text = texts[path]
        if Path(path).suffix.casefold() in {".py", ".pyw"}:
            try:
                tree = ast.parse(text, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                    signature = _python_signature(node)
                    symbols.append(SymbolInfo(node.name, kind, path, node.lineno, getattr(node, "end_lineno", node.lineno), signature, UnderstandingConfidence.HIGH, (UnderstandingEvidence("ast", path, f"{kind} {node.name}", node.lineno, getattr(node, "end_lineno", node.lineno), UnderstandingConfidence.HIGH),)))
        elif Path(path).suffix.casefold() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            symbols.extend(_javascript_symbols(path, text))
        if len(symbols) >= limit:
            return tuple(symbols[:limit])
    return tuple(symbols[:limit])


def _python_signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"{node.name}(... )"
    return f"class {getattr(node, 'name', 'unknown')}"


def _javascript_symbols(path: str, text: str) -> tuple[SymbolInfo, ...]:
    results: list[SymbolInfo] = []
    patterns = (
        (r"\bclass\s+([A-Za-z_$][\w$]*)", "class"),
        (r"\bfunction\s+([A-Za-z_$][\w$]*)\s*\(", "function"),
        (r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(", "function"),
        (r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)", "function"),
    )
    for pattern, kind in patterns:
        for match in re.finditer(pattern, text):
            line = text.count("\n", 0, match.start()) + 1
            results.append(SymbolInfo(match.group(1), kind, path, line, line, match.group(0), UnderstandingConfidence.MEDIUM, (UnderstandingEvidence("pattern", path, f"{kind} declaration", line, line, UnderstandingConfidence.MEDIUM),)))
    return tuple(sorted(results, key=lambda item: (item.line_start, item.name.casefold())))


def _analyze_references(texts: Mapping[str, str], symbols: Sequence[SymbolInfo], limit: int) -> tuple[ReferenceInfo, ...]:
    results: list[ReferenceInfo] = []
    symbol_names = tuple(dict.fromkeys(item.name for item in symbols))
    for path in sorted(texts, key=str.casefold):
        text = texts[path]
        if Path(path).suffix.casefold() in {".py", ".pyw"}:
            try:
                tree = ast.parse(text, filename=path)
            except SyntaxError:
                tree = None
            if tree is not None:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            results.append(_reference(path, alias.name, "imports", node.lineno, UnderstandingConfidence.HIGH))
                    elif isinstance(node, ast.ImportFrom):
                        module = "." * node.level + (node.module or "")
                        results.append(_reference(path, module, "imports", node.lineno, UnderstandingConfidence.HIGH))
        if Path(path).suffix.casefold() in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            for match in re.finditer(r"(?:from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]", text):
                line = text.count("\n", 0, match.start()) + 1
                results.append(_reference(path, match.group(1), "imports", line, UnderstandingConfidence.MEDIUM))
        for name in symbol_names:
            if path == next((item.path for item in symbols if item.name == name), None):
                continue
            match = re.search(rf"\b{re.escape(name)}\b", text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                results.append(_reference(path, name, "symbol_reference", line, UnderstandingConfidence.MEDIUM))
            if len(results) >= limit:
                return tuple(results[:limit])
    return tuple(results[:limit])


def _reference(path: str, target: str, relation: str, line: int | None, confidence: UnderstandingConfidence) -> ReferenceInfo:
    return ReferenceInfo(path, target, relation, line, confidence, (UnderstandingEvidence(relation, path, f"{relation}: {target}", line, line, confidence),))


def _analyze_dependencies(texts: Mapping[str, str], limit: int) -> tuple[DependencyInfo, ...]:
    results: list[DependencyInfo] = []
    for path in sorted(texts, key=str.casefold):
        text = texts[path]
        suffix = Path(path).suffix.casefold()
        if suffix in {".py", ".pyw"}:
            try:
                tree = ast.parse(text, filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        results.append(DependencyInfo(path, alias.name, "module_import", UnderstandingConfidence.HIGH, (UnderstandingEvidence("ast", path, f"import {alias.name}", node.lineno, node.lineno, UnderstandingConfidence.HIGH),)))
                elif isinstance(node, ast.ImportFrom):
                    module = "." * node.level + (node.module or "")
                    if module:
                        results.append(DependencyInfo(path, module, "module_import", UnderstandingConfidence.HIGH, (UnderstandingEvidence("ast", path, f"from {module} import ...", node.lineno, node.lineno, UnderstandingConfidence.HIGH),)))
        elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            for match in re.finditer(r"(?:from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]", text):
                line = text.count("\n", 0, match.start()) + 1
                target = match.group(1)
                results.append(DependencyInfo(path, target, "module_import", UnderstandingConfidence.MEDIUM, (UnderstandingEvidence("pattern", path, f"module import {target}", line, line, UnderstandingConfidence.MEDIUM),)))
        if len(results) >= limit:
            return tuple(results[:limit])
    return tuple(results[:limit])


def _root_evidence(structure: ProjectStructureResult, context: ProjectContext, relevant: Sequence[RelevantFile], symbols: Sequence[SymbolInfo], references: Sequence[ReferenceInfo], dependencies: Sequence[DependencyInfo], limit: int) -> tuple[UnderstandingEvidence, ...]:
    values: list[UnderstandingEvidence] = []
    for item in structure.frameworks[:16]:
        for evidence in item.evidence[:4]:
            values.append(UnderstandingEvidence("framework", evidence.split(":", 1)[0], evidence, confidence=UnderstandingConfidence(item.confidence if item.confidence in {"high", "medium", "low", "unknown"} else "medium")))
    for item in context.entry_points[:8]:
        values.append(UnderstandingEvidence("entry_point", item.name, "entry point detected by project structure tool", confidence=UnderstandingConfidence(item.confidence if item.confidence in {"high", "medium", "low", "unknown"} else "medium")))
    values.extend(item.evidence[0] for item in relevant[:24] if item.evidence)
    values.extend(item.evidence[0] for item in symbols[:24] if item.evidence)
    values.extend(item.evidence[0] for item in references[:24] if item.evidence)
    values.extend(item.evidence[0] for item in dependencies[:24] if item.evidence)
    return _bounded_evidence(values, limit)


def _bounded_evidence(values: Iterable[UnderstandingEvidence], limit: int) -> tuple[UnderstandingEvidence, ...]:
    unique: dict[tuple[str, str, str, int | None], UnderstandingEvidence] = {}
    for item in values:
        unique[(item.kind, item.path, item.detail, item.line_start)] = item
    return tuple(unique[key] for key in sorted(unique, key=lambda value: (value[1].casefold(), value[0], value[2]))[:limit])


def _components(structure: ProjectStructureResult, paths: Sequence[str], symbols: Sequence[SymbolInfo]) -> tuple[str, ...]:
    values: list[str] = []
    role_names = {"routes", "route", "api", "controllers", "controller", "views", "models", "model", "schemas", "schema", "serializers", "serializer", "services", "service", "repositories", "repository", "middleware", "auth", "authentication", "migrations", "tests", "config", "configuration"}
    for path in paths:
        for part in Path(path).parts:
            if part.casefold() in role_names:
                values.append(part)
    values.extend(item.kind for item in symbols if item.kind in {"class", "function", "async_function"})
    return _unique_text(values)[:64]


def _architecture(paths: Sequence[str], symbols: Sequence[SymbolInfo], dependencies: Sequence[DependencyInfo], limit: int) -> tuple[ArchitectureLayer, ...]:
    layer_paths: dict[str, list[str]] = {}
    for path in paths:
        role = _path_role(path)
        if role != "unknown":
            layer_paths.setdefault(role, []).append(path)
    layers: list[ArchitectureLayer] = []
    ordered = ("route", "controller", "middleware", "service", "repository", "model", "schema", "test", "configuration", "application")
    for name in ordered:
        values = tuple(sorted(dict.fromkeys(layer_paths.get(name, ())), key=str.casefold))
        if not values:
            continue
        relationships: list[str] = []
        if name in {"route", "controller"} and any(item.source in values and item.kind == "module_import" for item in dependencies):
            relationships.append("depends on imported modules observed in route/controller files")
        if name == "service" and any(_path_role(item.source) == "repository" for item in dependencies):
            relationships.append("service/repository relationship is partially evidenced")
        evidence = tuple(UnderstandingEvidence("architecture_path", path, f"path categorized as {name}", confidence=UnderstandingConfidence.MEDIUM) for path in values[:8])
        layers.append(ArchitectureLayer(name, values[:32], tuple(relationships), UnderstandingConfidence.MEDIUM, evidence))
    if not layers:
        return ()
    return tuple(layers[:limit])


def _path_role(path: str) -> str:
    text = path.casefold()
    name = Path(path).stem.casefold()
    for role, tokens in (
        ("test", ("test", "spec")),
        ("configuration", ("config", "settings", ".env", "pyproject", "package")),
        ("migration", ("migration", "migrations")),
        ("route", ("route", "routes", "router", "api", "endpoint")),
        ("controller", ("controller", "view", "views")),
        ("middleware", ("middleware", "auth", "security")),
        ("service", ("service", "services", "usecase")),
        ("repository", ("repository", "repositories", "dao")),
        ("model", ("model", "models", "orm")),
        ("schema", ("schema", "schemas", "serializer")),
        ("application", ("main", "app", "server", "wsgi", "asgi", "index")),
    ):
        if any(token in text or token in name for token in tokens):
            return role
    return "unknown"


def _is_test_path(path: str) -> bool:
    parts = {item.casefold() for item in Path(path).parts}
    return bool(parts & {"test", "tests", "spec", "specs"}) or Path(path).name.casefold().startswith("test_")


def _relevant_from_paths(paths: Sequence[str]) -> tuple[RelevantFile, ...]:
    return tuple(RelevantFile(path, _path_role(path), "medium", ("search evidence",), (UnderstandingEvidence("search_match", path, "matched the task-relevant search query"),)) for path in paths)


def _merge_symbols(existing: Sequence[SymbolInfo], additions: Sequence[SymbolInfo], limit: int) -> tuple[SymbolInfo, ...]:
    values: dict[tuple[str, str, str, int], SymbolInfo] = {(item.name, item.kind, item.path, item.line_start): item for item in existing}
    for item in additions:
        values[(item.name, item.kind, item.path, item.line_start)] = item
    return tuple(values[key] for key in sorted(values, key=lambda value: (value[2].casefold(), value[3], value[0].casefold()))[:limit])


def _understanding_confidence(structure: ProjectStructureResult, context: ProjectContext, evidence: Sequence[UnderstandingEvidence], symbols: Sequence[SymbolInfo]) -> UnderstandingConfidence:
    if not evidence:
        return UnderstandingConfidence.UNKNOWN
    if structure.truncated or context.truncated:
        return UnderstandingConfidence.MEDIUM
    if len(evidence) >= 8 and symbols:
        return UnderstandingConfidence.HIGH
    return UnderstandingConfidence.MEDIUM


__all__ = [
    "ArchitectureLayer",
    "CodebaseUnderstanding",
    "CodebaseUnderstandingBuilder",
    "DependencyInfo",
    "ReferenceInfo",
    "RelevantFile",
    "SymbolInfo",
    "UnderstandingCompleteness",
    "UnderstandingConfidence",
    "UnderstandingEvidence",
    "understand_codebase",
]
