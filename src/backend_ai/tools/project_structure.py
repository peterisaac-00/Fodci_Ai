"""Deterministic structural project detection for the Agent tool layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from backend_ai.tools.base import ToolError, ToolErrorCode, ToolMetadata
from backend_ai.tools.filesystem import (
    DEFAULT_IGNORED_DIRECTORIES,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_DIRECTORIES,
    DEFAULT_MAX_FILES,
    DiscoveredFile,
    FileDiscoveryResult,
    list_files,
)
from backend_ai.tools.read_file import read_file

DEFAULT_MAX_STRUCTURE_FILE_BYTES = 65_536
MAX_MAX_STRUCTURE_FILE_BYTES = 1_048_576
DEFAULT_MAX_INSPECTED_FILES = 64
MAX_MAX_INSPECTED_FILES = 256

SENSITIVE_FILE_NAMES = frozenset({".env"})
SENSITIVE_NAME_PARTS = ("credential", "secret", "private", "password")
SENSITIVE_SUFFIXES = (".pem", ".key", ".crt", ".p12", ".pfx")


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected technology backed by bounded filesystem evidence."""

    name: str
    confidence: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class LanguageSummary:
    """A deterministic language count derived from file extensions."""

    name: str
    files: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "files": self.files}


@dataclass(frozen=True, slots=True)
class DirectorySummary:
    """A major directory and its conservative heuristic category."""

    relative_path: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class ProjectStructureResult:
    """Immutable structural description of an explicitly selected project."""

    root: Path
    project_type: str
    frameworks: tuple[Detection, ...]
    languages: tuple[LanguageSummary, ...]
    package_managers: tuple[Detection, ...]
    databases: tuple[Detection, ...]
    test_frameworks: tuple[Detection, ...]
    infrastructure: tuple[Detection, ...]
    directories: tuple[DirectorySummary, ...]
    project_files: tuple[str, ...]
    important_files: tuple[str, ...]
    entry_points: tuple[Detection, ...]
    config_files: tuple[str, ...]
    dependency_files: tuple[str, ...]
    test_directories: tuple[str, ...]
    source_directories: tuple[str, ...]
    confidence: str
    evidence: tuple[str, ...]
    warnings: tuple[str, ...]
    truncated: bool
    truncation_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "project_type": self.project_type,
            "frameworks": [item.to_dict() for item in self.frameworks],
            "languages": [item.to_dict() for item in self.languages],
            "package_managers": [item.to_dict() for item in self.package_managers],
            "databases": [item.to_dict() for item in self.databases],
            "test_frameworks": [item.to_dict() for item in self.test_frameworks],
            "infrastructure": [item.to_dict() for item in self.infrastructure],
            "directories": [item.to_dict() for item in self.directories],
            "project_files": list(self.project_files),
            "important_files": list(self.important_files),
            "entry_points": [item.to_dict() for item in self.entry_points],
            "config_files": list(self.config_files),
            "dependency_files": list(self.dependency_files),
            "test_directories": list(self.test_directories),
            "source_directories": list(self.source_directories),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "truncation_reason": self.truncation_reason,
        }


class ProjectStructureTool:
    """First-class read-only Agent tool for structural project detection."""

    name = "project_structure"
    description = (
        "Detect likely project technologies, major components, languages, "
        "entry points, configuration, dependencies, tests, and infrastructure "
        "from bounded explicit filesystem evidence. Read-only and heuristic; "
        "not full project understanding."
    )
    metadata = ToolMetadata(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "required": ["project_root"],
            "properties": {
                "project_root": {"type": "string", "description": "Explicit project directory."},
                "max_files": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_FILES},
                "max_directories": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DIRECTORIES},
                "max_depth": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_DEPTH},
                "max_file_bytes": {"type": "integer", "minimum": 0, "default": DEFAULT_MAX_STRUCTURE_FILE_BYTES},
                "max_inspected_files": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_INSPECTED_FILES},
            },
        },
    )

    def run(self, arguments: Mapping[str, Any]) -> ProjectStructureResult:
        """Validate a structured request and detect project structure."""

        if not isinstance(arguments, Mapping):
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "project_structure arguments must be a mapping.")
        if "project_root" not in arguments:
            raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "project_structure requires 'project_root'.")
        return project_structure(
            arguments["project_root"],
            max_files=arguments.get("max_files", DEFAULT_MAX_FILES),
            max_directories=arguments.get("max_directories", DEFAULT_MAX_DIRECTORIES),
            max_depth=arguments.get("max_depth", DEFAULT_MAX_DEPTH),
            max_file_bytes=arguments.get("max_file_bytes", DEFAULT_MAX_STRUCTURE_FILE_BYTES),
            max_inspected_files=arguments.get("max_inspected_files", DEFAULT_MAX_INSPECTED_FILES),
        )


def project_structure(
    project_root: Path | str,
    *,
    max_files: int = DEFAULT_MAX_FILES,
    max_directories: int = DEFAULT_MAX_DIRECTORIES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_file_bytes: int = DEFAULT_MAX_STRUCTURE_FILE_BYTES,
    max_inspected_files: int = DEFAULT_MAX_INSPECTED_FILES,
) -> ProjectStructureResult:
    """Build an evidence-based structural description without project execution."""

    _validate_limits(max_file_bytes, max_inspected_files)
    inventory = list_files(
        project_root,
        max_files=max_files,
        max_directories=max_directories,
        max_depth=max_depth,
    )
    files = tuple(sorted(inventory.files, key=lambda item: item.relative_path.casefold()))
    directories = tuple(sorted(inventory.directories, key=lambda item: item.relative_path.casefold()))
    file_paths = tuple(item.relative_path for item in files if not _is_sensitive_path(item.relative_path))
    file_names = {path.casefold(): path for path in file_paths}
    directory_paths = tuple(item.relative_path for item in directories)

    important_files = tuple(sorted(
        path for path in file_paths if _is_important_file(path)
    ))
    dependency_files = tuple(sorted(
        path for path in file_paths if _is_dependency_file(path)
    ))
    config_files = tuple(sorted(
        path for path in file_paths if _is_config_file(path)
    ))
    test_directories = tuple(sorted(
        path for path in directory_paths if _directory_category(path) == "tests"
    ))
    source_directories = tuple(sorted(
        path for path in directory_paths if _directory_category(path) == "source"
    ))
    directory_results = tuple(
        DirectorySummary(path, _directory_category(path))
        for path in directory_paths
    )

    language_counts = _language_counts(files)
    languages = tuple(
        LanguageSummary(name, language_counts[name])
        for name in sorted(language_counts, key=str.casefold)
    )

    inspection_candidates = _inspection_candidates(
        files,
        dependency_files,
        config_files,
        file_names,
    )
    observed, warnings = _inspect_known_files(
        project_root=inventory.root,
        candidates=inspection_candidates,
        max_file_bytes=max_file_bytes,
        max_inspected_files=max_inspected_files,
    )
    warnings_set = set(warnings)
    if any(_is_sensitive_path(item.relative_path) for item in files):
        warnings_set.add("Sensitive files were excluded from structural content inspection.")
    if inventory.truncated:
        warnings_set.add("File discovery reached a configured limit; structure may be incomplete.")

    frameworks = _detect_frameworks(files, observed)
    package_managers = _detect_package_managers(dependency_files, observed)
    databases = _detect_databases(files, observed)
    test_frameworks = _detect_test_frameworks(files, test_directories, observed)
    infrastructure = _detect_infrastructure(file_paths)
    entry_points = _detect_entry_points(files, observed)
    project_type = _project_type(languages, frameworks)
    evidence = _root_evidence(
        files=files,
        important_files=important_files,
        languages=languages,
        frameworks=frameworks,
        package_managers=package_managers,
        test_frameworks=test_frameworks,
        infrastructure=infrastructure,
    )
    confidence = _overall_confidence(project_type, evidence, inventory.truncated)
    warnings_set.update(_detection_warnings(project_type, files, evidence))

    return ProjectStructureResult(
        root=inventory.root,
        project_type=project_type,
        frameworks=frameworks,
        languages=languages,
        package_managers=package_managers,
        databases=databases,
        test_frameworks=test_frameworks,
        infrastructure=infrastructure,
        directories=directory_results,
        project_files=file_paths,
        important_files=important_files,
        entry_points=entry_points,
        config_files=config_files,
        dependency_files=dependency_files,
        test_directories=test_directories,
        source_directories=source_directories,
        confidence=confidence,
        evidence=evidence,
        warnings=tuple(sorted(warnings_set)),
        truncated=inventory.truncated,
        truncation_reason=inventory.truncation_reason,
    )


def _validate_limits(max_file_bytes: int, max_inspected_files: int) -> None:
    if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or max_file_bytes < 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_file_bytes must be a non-negative integer.")
    if max_file_bytes > MAX_MAX_STRUCTURE_FILE_BYTES:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"max_file_bytes cannot exceed {MAX_MAX_STRUCTURE_FILE_BYTES}.",
        )
    if not isinstance(max_inspected_files, int) or isinstance(max_inspected_files, bool) or max_inspected_files <= 0:
        raise ToolError(ToolErrorCode.INVALID_ARGUMENT, "max_inspected_files must be a positive integer.")
    if max_inspected_files > MAX_MAX_INSPECTED_FILES:
        raise ToolError(
            ToolErrorCode.INVALID_ARGUMENT,
            f"max_inspected_files cannot exceed {MAX_MAX_INSPECTED_FILES}.",
        )


def _is_sensitive_path(relative_path: str) -> bool:
    for part in Path(relative_path).parts:
        lowered = part.casefold()
        if lowered in SENSITIVE_FILE_NAMES:
            return True
        if any(token in lowered for token in SENSITIVE_NAME_PARTS):
            return True
        if lowered.endswith(SENSITIVE_SUFFIXES):
            return True
    return False


def _basename(relative_path: str) -> str:
    return Path(relative_path).name.casefold()


def _is_important_file(relative_path: str) -> bool:
    name = _basename(relative_path)
    suffix = Path(name).suffix
    return (
        name.startswith("readme")
        or name in {
            "pyproject.toml", "requirements.txt", "package.json", "package-lock.json",
            "pnpm-lock.yaml", "yarn.lock", "bun.lock", "tsconfig.json", "dockerfile",
            "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
            ".env.example", ".gitignore", "makefile", "manage.py", "pipfile", "poetry.lock",
            "uv.lock", "pytest.ini", "tox.ini", ".pre-commit-config.yaml", ".gitlab-ci.yml",
            "azure-pipelines.yml",
        }
        or ("/.github/workflows/" in f"/{relative_path.casefold()}/" and suffix in {".yml", ".yaml"})
        or name.startswith(("jest.config", "vitest.config"))
    )


def _is_dependency_file(relative_path: str) -> bool:
    name = _basename(relative_path)
    return (
        name.startswith("requirements") and name.endswith(".txt")
        or name in {
            "pyproject.toml", "setup.py", "setup.cfg", "pipfile", "pipfile.lock", "poetry.lock",
            "uv.lock", "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock",
        }
    )


def _is_config_file(relative_path: str) -> bool:
    name = _basename(relative_path)
    return (
        name in {
            "pyproject.toml", "setup.cfg", "tsconfig.json", "pytest.ini", "tox.ini", ".flake8", ".env.example",
            ".pre-commit-config.yaml", "docker-compose.yml", "docker-compose.yaml", "compose.yml",
            "compose.yaml", ".gitlab-ci.yml", "azure-pipelines.yml",
        }
        or name.startswith(("jest.config", "vitest.config"))
        or ("/.github/workflows/" in f"/{relative_path.casefold()}/" and name.endswith((".yml", ".yaml")))
    )


def _directory_category(relative_path: str) -> str:
    name = Path(relative_path).name.casefold()
    if name in {"tests", "test", "spec", "specs"}:
        return "tests"
    if name in {"src", "app", "api", "backend", "frontend", "services", "workers", "modules"}:
        return "source"
    if name in {"docs", "doc", "documentation"}:
        return "documentation"
    if name in {"migrations", "migration", "database", "databases", "db"}:
        return "database"
    if name in {"config", "configs", "configuration"}:
        return "configuration"
    if name in {"scripts", "tools"}:
        return "scripts"
    return "other"


def _language_counts(files: tuple[DiscoveredFile, ...]) -> dict[str, int]:
    extension_map = {
        ".py": "Python", ".pyw": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
        ".ts": "TypeScript", ".tsx": "TypeScript", ".sql": "SQL", ".json": "JSON",
        ".yaml": "YAML", ".yml": "YAML", ".md": "Markdown", ".markdown": "Markdown",
        ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "CSS", ".sh": "Shell",
        ".bash": "Shell",
    }
    counts: dict[str, int] = {}
    for item in files:
        if _is_sensitive_path(item.relative_path):
            continue
        language = extension_map.get(item.extension.casefold())
        if language is not None:
            counts[language] = counts.get(language, 0) + 1
        elif item.name.casefold() == "makefile":
            counts["Shell"] = counts.get("Shell", 0) + 1
        elif item.name.casefold() == "dockerfile":
            counts["Dockerfile"] = counts.get("Dockerfile", 0) + 1
    return counts


def _inspection_candidates(
    files: tuple[DiscoveredFile, ...],
    dependency_files: tuple[str, ...],
    config_files: tuple[str, ...],
    file_names: dict[str, str],
) -> tuple[str, ...]:
    candidates = set(dependency_files) | set(config_files)
    for item in files:
        name = item.name.casefold()
        suffix = item.extension.casefold()
        if name in {"manage.py", "main.py", "app.py", "server.py", "wsgi.py", "asgi.py", "index.js", "server.js", "app.js", "index.ts", "server.ts"}:
            if suffix in {".py", ".js", ".ts", ".tsx", ""}:
                candidates.add(item.relative_path)
        if name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts")):
            candidates.add(item.relative_path)
    return tuple(sorted(candidates, key=str.casefold))


def _inspect_known_files(
    project_root: Path,
    candidates: tuple[str, ...],
    max_file_bytes: int,
    max_inspected_files: int,
) -> tuple[dict[str, str], tuple[str, ...]]:
    observed: dict[str, str] = {}
    warnings: set[str] = set()
    for relative_path in candidates[:max_inspected_files]:
        if _is_sensitive_path(relative_path):
            warnings.add("Sensitive files were excluded from structural content inspection.")
            continue
        try:
            result = read_file(project_root, relative_path, max_bytes=max_file_bytes)
        except ToolError as exc:
            if exc.code == ToolErrorCode.FILE_TOO_LARGE:
                warnings.add("Some known files exceeded the structural inspection byte limit.")
            elif exc.code == ToolErrorCode.INVALID_UTF8:
                warnings.add("Some known files were not valid UTF-8 and were not inspected.")
            else:
                warnings.add(f"Some known files could not be inspected: {exc.code.value}.")
            continue
        observed[relative_path] = result.content
    if len(candidates) > max_inspected_files:
        warnings.add("Structural inspection reached max_inspected_files; evidence may be incomplete.")
    return observed, tuple(sorted(warnings))


def _is_test_path(relative_path: str) -> bool:
    parts = {part.casefold() for part in Path(relative_path).parts}
    return bool(parts & {"test", "tests", "spec", "specs"}) or _is_test_file(relative_path)


def _focused_texts(texts: dict[str, str]) -> dict[str, str]:
    entry_names = {
        "manage.py", "main.py", "app.py", "server.py", "wsgi.py", "asgi.py",
        "index.js", "server.js", "app.js", "index.ts", "server.ts",
        "db.py", "database.py", "models.py", "settings.py",
    }
    return {
        path: text
        for path, text in texts.items()
        if (_is_dependency_file(path) or _is_config_file(path) or _basename(path) in entry_names)
        or (not _is_test_path(path) and Path(path).suffix.casefold() in {".py", ".js", ".jsx", ".ts", ".tsx"})
    }


def _contains(texts: dict[str, str], pattern: str) -> tuple[str, ...]:
    compiled = re.compile(pattern, re.IGNORECASE)
    return tuple(
        f"{path}: matched structural evidence"
        for path in sorted(texts, key=str.casefold)
        if compiled.search(texts[path])
    )


def _dependency_evidence(texts: dict[str, str], tokens: tuple[str, ...]) -> tuple[str, ...]:
    pattern = re.compile("|".join(re.escape(token) for token in tokens), re.IGNORECASE)
    return tuple(
        f"{path}: dependency/configuration evidence"
        for path in sorted(texts, key=str.casefold)
        if pattern.search(texts[path])
    )


def _make_detection(name: str, evidence: tuple[str, ...]) -> Detection | None:
    if not evidence:
        return None
    confidence = "high" if len(evidence) >= 2 else "medium"
    return Detection(name, confidence, tuple(sorted(set(evidence), key=str.casefold)))


def _detect_frameworks(files: tuple[DiscoveredFile, ...], texts: dict[str, str]) -> tuple[Detection, ...]:
    paths = {item.relative_path: item for item in files if not _is_sensitive_path(item.relative_path)}
    focused_texts = _focused_texts(texts)
    py_files = [path for path, item in paths.items() if item.extension.casefold() in {".py", ".pyw"}]
    js_files = [path for path, item in paths.items() if item.extension.casefold() in {".js", ".jsx", ".ts", ".tsx"}]
    detections: list[Detection] = []
    python = _make_detection("Python", tuple(f"{path}: Python source file" for path in sorted(py_files)[:3]))
    if python:
        detections.append(python)
    node_evidence = tuple(f"{path}: Node project file" for path in sorted(paths) if _basename(path) in {"package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock"})
    node_evidence += tuple(f"{path}: JavaScript/TypeScript source file" for path in sorted(js_files)[:2])
    node = _make_detection("Node.js", node_evidence)
    if node:
        detections.append(node)
    for name, tokens, source_pattern in (
        ("Django", ("django",), r"\bmanage\.py\b|\bdjango\b"),
        ("FastAPI", ("fastapi",), r"(?:from|import)\s+fastapi|FastAPI\s*\("),
        ("Flask", ("flask",), r"(?:from|import)\s+flask|Flask\s*\("),
        ("Express", ("express",), r"require\s*\(\s*['\"]express|from\s+['\"]express"),
        ("React", ("react",), r"from\s+['\"]react|require\s*\(\s*['\"]react|React\."),
    ):
        evidence = _dependency_evidence(focused_texts, tokens) + _contains(focused_texts, source_pattern)
        if name == "Django" and any(_basename(path) == "manage.py" for path in paths):
            evidence += tuple(f"{path}: manage.py present" for path in sorted(paths) if _basename(path) == "manage.py")
        detection = _make_detection(name, evidence)
        if detection:
            detections.append(detection)
    if any(item.extension.casefold() in {".ts", ".tsx"} for item in paths.values()) or "tsconfig.json" in {_basename(path) for path in paths}:
        evidence = tuple(f"{path}: TypeScript evidence" for path, item in sorted(paths.items()) if item.extension.casefold() in {".ts", ".tsx"})
        evidence += tuple(f"{path}: tsconfig.json present" for path in sorted(paths) if _basename(path) == "tsconfig.json")
        detection = _make_detection("TypeScript", evidence)
        if detection:
            detections.append(detection)
    if any(item.extension.casefold() in {".js", ".jsx"} for item in paths.values()):
        detection = _make_detection("JavaScript", tuple(f"{path}: JavaScript source file" for path, item in sorted(paths.items()) if item.extension.casefold() in {".js", ".jsx"}))
        if detection:
            detections.append(detection)
    return tuple(sorted(detections, key=lambda item: item.name.casefold()))


def _detect_package_managers(dependency_files: tuple[str, ...], texts: dict[str, str]) -> tuple[Detection, ...]:
    names: dict[str, list[str]] = {}
    for path in dependency_files:
        name = _basename(path)
        if name == "package-lock.json":
            names.setdefault("npm", []).append(f"{path}: package-lock.json present")
        elif name == "package.json":
            names.setdefault("npm", []).append(f"{path}: package.json present")
        elif name == "pnpm-lock.yaml":
            names.setdefault("pnpm", []).append(f"{path}: pnpm-lock.yaml present")
        elif name == "yarn.lock":
            names.setdefault("yarn", []).append(f"{path}: yarn.lock present")
        elif name == "bun.lock":
            names.setdefault("bun", []).append(f"{path}: bun.lock present")
        elif name in {"pipfile", "pipfile.lock"}:
            names.setdefault("pipenv", []).append(f"{path}: Pipfile present")
        elif name == "poetry.lock" or (name == "pyproject.toml" and re.search(r"\[tool\.poetry\]", texts.get(path, ""), re.IGNORECASE)):
            names.setdefault("poetry", []).append(f"{path}: Poetry configuration present")
        elif name == "uv.lock" or (name == "pyproject.toml" and re.search(r"\[tool\.uv\]", texts.get(path, ""), re.IGNORECASE)):
            names.setdefault("uv", []).append(f"{path}: uv configuration present")
        elif name.startswith("requirements") and name.endswith(".txt"):
            names.setdefault("pip", []).append(f"{path}: requirements file present")
    return tuple(
        Detection(name, "high" if len(evidence) >= 2 else "medium", tuple(sorted(evidence, key=str.casefold)))
        for name, evidence in sorted(names.items(), key=lambda item: item[0].casefold())
    )


def _detect_databases(files: tuple[DiscoveredFile, ...], texts: dict[str, str]) -> tuple[Detection, ...]:
    focused_texts = _focused_texts(texts)
    patterns = {
        "PostgreSQL": ("psycopg", "psycopg2", "asyncpg", "postgresql", "postgres"),
        "MySQL": ("mysqlclient", "pymysql", "mysql"),
        "MariaDB": ("mariadb",),
        "SQLite": ("sqlite3", "sqlite"),
        "MongoDB": ("pymongo", "motor", "mongoose", "mongodb"),
    }
    detections: list[Detection] = []
    for name, tokens in patterns.items():
        evidence = _dependency_evidence(focused_texts, tokens)
        if name == "SQLite":
            evidence += tuple(f"{item.relative_path}: database file extension" for item in files if item.extension.casefold() in {".sqlite", ".sqlite3", ".db"})
        detection = _make_detection(name, evidence)
        if detection:
            detections.append(detection)
    return tuple(sorted(detections, key=lambda item: item.name.casefold()))


def _source_import_evidence(text: str, module: str) -> bool:
    import_prefixes = (f"import {module}", f"from {module} import")
    require_patterns = (f'require("{module}")', f"require('{module}')")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(import_prefixes) or any(pattern in stripped for pattern in require_patterns):
            return True
    return False


def _detect_test_frameworks(files: tuple[DiscoveredFile, ...], test_directories: tuple[str, ...], texts: dict[str, str]) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    test_texts = {
        path: text
        for path, text in texts.items()
        if _is_dependency_file(path) or _is_config_file(path) or _is_test_path(path)
    }
    pytest_evidence = tuple(
        f"{path}: pytest evidence"
        for path, text in test_texts.items()
        if (_is_dependency_file(path) and re.search(r"\bpytest\b", text, re.IGNORECASE))
        or _source_import_evidence(text, "pytest")
    )
    pytest_evidence += tuple(f"{path}: pytest configuration" for path in files_to_paths(files) if _basename(path) in {"pytest.ini", "tox.ini"})
    unittest_evidence = tuple(f"{path}: unittest import/evidence" for path, text in test_texts.items() if _source_import_evidence(text, "unittest"))
    jest_evidence = tuple(
        f"{path}: Jest evidence"
        for path, text in test_texts.items()
        if (_is_dependency_file(path) and re.search(r"[\"']jest[\"']\s*:", text, re.IGNORECASE))
        or _source_import_evidence(text, "jest")
    )
    vitest_evidence = tuple(
        f"{path}: Vitest evidence"
        for path, text in test_texts.items()
        if (_is_dependency_file(path) and re.search(r"[\"']vitest[\"']\s*:", text, re.IGNORECASE))
        or _source_import_evidence(text, "vitest")
    )
    for name, evidence in (("pytest", pytest_evidence), ("unittest", unittest_evidence), ("Jest", jest_evidence), ("Vitest", vitest_evidence)):
        detection = _make_detection(name, evidence)
        if detection:
            detections.append(detection)
    generic_evidence = tuple(f"{path}: test directory" for path in test_directories)
    generic_evidence += tuple(f"{item.relative_path}: test file naming" for item in files if _is_test_file(item.relative_path))
    generic = _make_detection("generic tests", generic_evidence)
    if generic:
        detections.append(generic)
    return tuple(sorted(detections, key=lambda item: item.name.casefold()))


def _detect_infrastructure(file_paths: tuple[str, ...]) -> tuple[Detection, ...]:
    docker = tuple(f"{path}: Dockerfile present" for path in file_paths if _basename(path) == "dockerfile")
    compose = tuple(f"{path}: Compose file present" for path in file_paths if _basename(path) in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"})
    ci = tuple(f"{path}: CI configuration present" for path in file_paths if "/.github/workflows/" in f"/{path.casefold()}/" or _basename(path) in {".gitlab-ci.yml", "azure-pipelines.yml"})
    detections: list[Detection] = []
    for name, evidence in (("Docker", docker), ("Docker Compose", compose), ("CI", ci)):
        detection = _make_detection(name, evidence)
        if detection:
            detections.append(detection)
    return tuple(sorted(detections, key=lambda item: item.name.casefold()))


def _detect_entry_points(files: tuple[DiscoveredFile, ...], texts: dict[str, str]) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    known = {"manage.py", "main.py", "app.py", "wsgi.py", "asgi.py", "server.py", "server.js", "app.js", "index.js", "index.ts", "server.ts"}
    for item in files:
        if _basename(item.relative_path) in known:
            confidence = "high" if _basename(item.relative_path) in {"manage.py", "wsgi.py", "asgi.py"} else "medium"
            detections.append(Detection(item.relative_path, confidence, (f"{item.relative_path}: conventional entry-point filename",)))
    package = next((text for path, text in texts.items() if _basename(path) == "package.json"), None)
    if package:
        try:
            parsed = json.loads(package)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            main = parsed.get("main")
            if isinstance(main, str) and main:
                detections.append(Detection(main, "high", ("package.json: main field",)))
            bin_value = parsed.get("bin")
            if isinstance(bin_value, str) and bin_value:
                detections.append(Detection(bin_value, "high", ("package.json: bin field",)))
            elif isinstance(bin_value, dict):
                for value in bin_value.values():
                    if isinstance(value, str) and value:
                        detections.append(Detection(value, "high", ("package.json: bin field",)))
    unique = {(item.name, item.confidence, item.evidence): item for item in detections}
    return tuple(sorted(unique.values(), key=lambda item: item.name.casefold()))


def _project_type(languages: tuple[LanguageSummary, ...], frameworks: tuple[Detection, ...]) -> str:
    language_names = {item.name for item in languages}
    framework_names = {item.name for item in frameworks}
    if not language_names and not framework_names:
        return "empty"
    has_python = "Python" in language_names or bool(framework_names & {"Django", "FastAPI", "Flask"})
    has_node = bool(language_names & {"JavaScript", "TypeScript"}) or bool(framework_names & {"Node.js", "Express", "React"})
    if has_python and has_node:
        return "mixed"
    if has_python:
        return "python"
    if has_node:
        return "node"
    return "other"


def _root_evidence(
    files: tuple[DiscoveredFile, ...],
    important_files: tuple[str, ...],
    languages: tuple[LanguageSummary, ...],
    frameworks: tuple[Detection, ...],
    package_managers: tuple[Detection, ...],
    test_frameworks: tuple[Detection, ...],
    infrastructure: tuple[Detection, ...],
) -> tuple[str, ...]:
    evidence: set[str] = set(important_files)
    evidence.update(item.relative_path for item in files if item.extension.casefold() in {".py", ".js", ".ts", ".tsx"})
    for collection in (frameworks, package_managers, test_frameworks, infrastructure):
        for item in collection:
            evidence.update(item.evidence)
    return tuple(sorted(evidence, key=str.casefold))


def _overall_confidence(project_type: str, evidence: tuple[str, ...], truncated: bool) -> str:
    if project_type == "empty" or not evidence:
        return "low"
    if truncated:
        return "medium"
    if len(evidence) >= 3:
        return "high"
    return "medium"


def _detection_warnings(project_type: str, files: tuple[DiscoveredFile, ...], evidence: tuple[str, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    if project_type == "empty":
        warnings.append("No project files were discovered.")
    elif not evidence:
        warnings.append("No strong structural evidence was detected.")
    return tuple(warnings)


def files_to_paths(files: tuple[DiscoveredFile, ...]) -> tuple[str, ...]:
    return tuple(item.relative_path for item in files)


def _is_test_file(relative_path: str) -> bool:
    name = _basename(relative_path)
    return name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.js", ".test.ts", ".spec.ts"))


__all__ = [
    "DEFAULT_MAX_INSPECTED_FILES",
    "DEFAULT_MAX_STRUCTURE_FILE_BYTES",
    "Detection",
    "DirectorySummary",
    "LanguageSummary",
    "MAX_MAX_INSPECTED_FILES",
    "MAX_MAX_STRUCTURE_FILE_BYTES",
    "ProjectStructureResult",
    "ProjectStructureTool",
    "project_structure",
]
