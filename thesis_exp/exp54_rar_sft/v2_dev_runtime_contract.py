"""Complete source and dependency identity for formal Exp54 V2 dev."""

from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
import platform
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any

from thesis_exp.exp54_rar_sft import REPO_ROOT


V2_RUNTIME_SOURCE_PATHS = {
    "thesis_exp/__init__.py": REPO_ROOT / "thesis_exp/__init__.py",
    "thesis_exp/src/__init__.py": REPO_ROOT / "thesis_exp/src/__init__.py",
    "thesis_exp/src/edujudge/__init__.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/__init__.py"
    ),
    "thesis_exp/src/edujudge/exp02/__init__.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/exp02/__init__.py"
    ),
    "thesis_exp/src/edujudge/utils/__init__.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/__init__.py"
    ),
    "thesis_exp/exp54_rar_sft/__init__.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/__init__.py"
    ),
    "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py"
    ),
    "thesis_exp/src/edujudge/utils/io.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/io.py"
    ),
    "thesis_exp/src/edujudge/utils/text_norm.py": (
        REPO_ROOT / "thesis_exp/src/edujudge/utils/text_norm.py"
    ),
    "thesis_exp/exp54_rar_sft/authorization_guard.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/authorization_guard.py"
    ),
    "thesis_exp/exp54_rar_sft/audit_rar0_alignment.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/audit_rar0_alignment.py"
    ),
    "thesis_exp/exp54_rar_sft/block_loss.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/block_loss.py"
    ),
    "thesis_exp/exp54_rar_sft/training_contract.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/training_contract.py"
    ),
    "thesis_exp/exp54_rar_sft/inference_contract.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/inference_contract.py"
    ),
    "thesis_exp/exp54_rar_sft/train_rar_sft.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/train_rar_sft.py"
    ),
    "thesis_exp/exp54_rar_sft/run_dev_inference.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/run_dev_inference.py"
    ),
    "thesis_exp/exp54_rar_sft/structured_decoder_v2.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/structured_decoder_v2.py"
    ),
    "thesis_exp/exp54_rar_sft/v2_dev_runtime_contract.py": Path(__file__),
    "thesis_exp/exp54_rar_sft/v2_dev_authorization_guard.py": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/v2_dev_authorization_guard.py"
    ),
    "thesis_exp/exp54_rar_sft/run_dev_inference_v2.py": (
        REPO_ROOT / "thesis_exp/exp54_rar_sft/run_dev_inference_v2.py"
    ),
    "thesis_exp/scripts/run_exp54_v2_dev_campaign.sh": (
        REPO_ROOT / "thesis_exp/scripts/run_exp54_v2_dev_campaign.sh"
    ),
    "thesis_exp/exp54_rar_sft/configs/"
    "inference_protocol_v2_candidate.json": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/"
        "inference_protocol_v2_candidate.json"
    ),
    "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf"
    ),
    "thesis_exp/exp54_rar_sft/configs/"
    "canonical_rubric_registry.json": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/"
        "canonical_rubric_registry.json"
    ),
    "thesis_exp/exp54_rar_sft/configs/"
    "training_configuration_candidate.json": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/"
        "training_configuration_candidate.json"
    ),
    "thesis_exp/exp54_rar_sft/configs/"
    "qwen_tokenizer_lock_spec.json": (
        REPO_ROOT
        / "thesis_exp/exp54_rar_sft/configs/qwen_tokenizer_lock_spec.json"
    ),
}
EXPECTED_V2_RUNTIME_SOURCE_NAMES = frozenset(
    {
        "thesis_exp/__init__.py",
        "thesis_exp/src/__init__.py",
        "thesis_exp/src/edujudge/__init__.py",
        "thesis_exp/src/edujudge/exp02/__init__.py",
        "thesis_exp/src/edujudge/utils/__init__.py",
        "thesis_exp/exp54_rar_sft/__init__.py",
        "thesis_exp/src/edujudge/exp02/build_exp02_dataset.py",
        "thesis_exp/src/edujudge/utils/io.py",
        "thesis_exp/src/edujudge/utils/text_norm.py",
        "thesis_exp/exp54_rar_sft/authorization_guard.py",
        "thesis_exp/exp54_rar_sft/audit_rar0_alignment.py",
        "thesis_exp/exp54_rar_sft/block_loss.py",
        "thesis_exp/exp54_rar_sft/training_contract.py",
        "thesis_exp/exp54_rar_sft/inference_contract.py",
        "thesis_exp/exp54_rar_sft/train_rar_sft.py",
        "thesis_exp/exp54_rar_sft/run_dev_inference.py",
        "thesis_exp/exp54_rar_sft/structured_decoder_v2.py",
        "thesis_exp/exp54_rar_sft/v2_dev_runtime_contract.py",
        "thesis_exp/exp54_rar_sft/v2_dev_authorization_guard.py",
        "thesis_exp/exp54_rar_sft/run_dev_inference_v2.py",
        "thesis_exp/scripts/run_exp54_v2_dev_campaign.sh",
        "thesis_exp/exp54_rar_sft/configs/"
        "inference_protocol_v2_candidate.json",
        "thesis_exp/exp54_rar_sft/configs/rar_sft_output_v2.ebnf",
        "thesis_exp/exp54_rar_sft/configs/"
        "canonical_rubric_registry.json",
        "thesis_exp/exp54_rar_sft/configs/"
        "training_configuration_candidate.json",
        "thesis_exp/exp54_rar_sft/configs/"
        "qwen_tokenizer_lock_spec.json",
    }
)
EXPECTED_V2_RUNTIME_SOURCE_COUNT = 26
RUNTIME_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "xgrammar": "xgrammar",
    "apache-tvm-ffi": "apache-tvm-ffi",
    "numpy": "numpy",
    "peft": "peft",
    "safetensors": "safetensors",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def v2_runtime_source_closure() -> dict[str, str]:
    if set(V2_RUNTIME_SOURCE_PATHS) != EXPECTED_V2_RUNTIME_SOURCE_NAMES:
        raise RuntimeError("V2 runtime source-name set differs")
    if len(V2_RUNTIME_SOURCE_PATHS) != EXPECTED_V2_RUNTIME_SOURCE_COUNT:
        raise RuntimeError("V2 runtime source count differs")
    missing = [
        str(path)
        for path in V2_RUNTIME_SOURCE_PATHS.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"V2 runtime source closure is incomplete: {missing}"
        )
    return {
        name: sha256_file(path)
        for name, path in sorted(V2_RUNTIME_SOURCE_PATHS.items())
    }


def v2_runtime_source_closure_sha256(
    closure: dict[str, str] | None = None,
) -> str:
    resolved = closure or v2_runtime_source_closure()
    if set(resolved) != EXPECTED_V2_RUNTIME_SOURCE_NAMES:
        raise ValueError("V2 runtime closure keys differ")
    return canonical_sha256(sorted(resolved.items()))


def observed_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            field: importlib.metadata.version(distribution)
            for field, distribution in RUNTIME_PACKAGES.items()
        },
    }


def expected_runtime_versions(
    protocol: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, str]:
    return {
        "python": str(protocol["runtime"]["python"]),
        "torch": str(protocol["runtime"]["torch"]),
        "transformers": str(protocol["runtime"]["transformers"]),
        "tokenizers": str(protocol["runtime"]["tokenizers"]),
        "xgrammar": str(protocol["backend"]["version"]),
        "apache-tvm-ffi": str(
            protocol["backend"]["dependency_wheels"][0]["version"]
        ),
        "numpy": str(training_config["runtime"]["numpy"]),
        "peft": str(training_config["runtime"]["peft"]),
        "safetensors": str(training_config["runtime"]["safetensors"]),
    }


def require_runtime_versions(
    protocol: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, str]:
    expected = expected_runtime_versions(protocol, training_config)
    actual = observed_runtime_versions()
    if actual != expected:
        raise PermissionError(
            f"V2 runtime versions differ: {actual} != {expected}"
        )
    return actual


def observed_installed_distribution(
    distribution_name: str,
) -> dict[str, Any]:
    distribution = importlib.metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise PermissionError(
            f"{distribution_name}: installed file list is unavailable"
        )
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = (
        json.loads(direct_url_text)
        if direct_url_text is not None
        else None
    )
    if isinstance(direct_url, dict):
        if "dir_info" in direct_url:
            raise PermissionError(
                f"{distribution_name}: source-directory install is prohibited"
            )
        if "vcs_info" in direct_url:
            raise PermissionError(
                f"{distribution_name}: VCS install is prohibited"
            )
    entries = []
    for relative in sorted(str(item) for item in files):
        path = Path(distribution.locate_file(relative))
        if not path.is_file() or path.is_symlink():
            raise PermissionError(
                f"{distribution_name}: distribution file is unavailable: "
                f"{relative}"
            )
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "file_count": len(entries),
        "file_tree_sha256": canonical_sha256(entries),
        "direct_url_sha256": (
            hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest()
            if direct_url_text is not None
            else None
        ),
    }


_INSTALLER_GENERATED_SUFFIXES = {
    "RECORD",
    "INSTALLER",
    "direct_url.json",
    "REQUESTED",
}


def _is_installer_generated(relative: str) -> bool:
    parts = Path(relative).parts
    return (
        relative.endswith(".pyc")
        or "__pycache__" in parts
        or (
            parts
            and parts[0] == ".."
            and "bin" in parts
        )
        or (
            any(part.endswith(".dist-info") for part in parts)
            and parts[-1] in _INSTALLER_GENERATED_SUFFIXES
        )
    )


def _wheel_install_relative(relative: str) -> str:
    parts = Path(relative).parts
    for marker in ("purelib", "platlib"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0 and parts[index - 1].endswith(".data"):
                return str(Path(*parts[index + 1 :]))
    return str(Path(*parts))


def _tree_entry(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_installed_distribution_matches_wheel(
    distribution_name: str,
    wheel_path: Path,
    *,
    expected_wheel_sha256: str,
) -> dict[str, Any]:
    """Bind an installed distribution byte-for-byte to a reviewed wheel."""
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise PermissionError(
            f"{distribution_name}: reviewed wheel is unavailable"
        )
    wheel_bytes = wheel_path.read_bytes()
    if hashlib.sha256(wheel_bytes).hexdigest() != expected_wheel_sha256:
        raise PermissionError(
            f"{distribution_name}: reviewed wheel SHA-256 differs"
        )
    distribution = importlib.metadata.distribution(distribution_name)
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = (
        json.loads(direct_url_text)
        if direct_url_text is not None
        else None
    )
    if isinstance(direct_url, dict):
        if "dir_info" in direct_url:
            raise PermissionError(
                f"{distribution_name}: source-directory install is prohibited"
            )
        if "vcs_info" in direct_url:
            raise PermissionError(
                f"{distribution_name}: VCS install is prohibited"
            )
        archive = direct_url.get("archive_info")
        if not isinstance(archive, dict):
            raise PermissionError(
                f"{distribution_name}: archive provenance is incomplete"
            )
        archive_hash = archive.get("hash")
        if archive_hash != f"sha256={expected_wheel_sha256}":
            raise PermissionError(
                f"{distribution_name}: installed archive hash differs"
            )
        url_name = Path(
            urllib.parse.unquote(
                urllib.parse.urlparse(str(direct_url.get("url") or "")).path
            )
        ).name
        if url_name != wheel_path.name:
            raise PermissionError(
                f"{distribution_name}: installed archive filename differs"
            )
    wheel_entries: list[dict[str, Any]] = []
    wheel_payload: dict[str, tuple[int, str]] = {}
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            if info.is_dir():
                continue
            relative = _wheel_install_relative(info.filename)
            if _is_installer_generated(relative):
                continue
            payload = archive.read(info)
            entry = _tree_entry(relative, payload)
            if relative in wheel_payload:
                raise PermissionError(
                    f"{distribution_name}: duplicate wheel payload path"
                )
            wheel_payload[relative] = (entry["size"], entry["sha256"])
            wheel_entries.append(entry)
    files = distribution.files
    if files is None:
        raise PermissionError(
            f"{distribution_name}: installed file list is unavailable"
        )
    installed_entries: list[dict[str, Any]] = []
    installed_payload: dict[str, tuple[int, str]] = {}
    for item in sorted(str(value) for value in files):
        relative = str(Path(item))
        if _is_installer_generated(relative):
            continue
        path = Path(distribution.locate_file(relative))
        if path.is_symlink() or not path.is_file():
            raise PermissionError(
                f"{distribution_name}: installed payload is unavailable"
            )
        payload = path.read_bytes()
        entry = _tree_entry(relative, payload)
        installed_payload[relative] = (entry["size"], entry["sha256"])
        installed_entries.append(entry)
    if installed_payload != wheel_payload:
        missing = sorted(set(wheel_payload) - set(installed_payload))
        extra = sorted(set(installed_payload) - set(wheel_payload))
        changed = sorted(
            path
            for path in set(wheel_payload) & set(installed_payload)
            if wheel_payload[path] != installed_payload[path]
        )
        raise PermissionError(
            f"{distribution_name}: installed payload differs from reviewed "
            f"wheel (missing={missing[:3]}, extra={extra[:3]}, "
            f"changed={changed[:3]})"
        )
    return {
        "distribution": distribution_name,
        "version": distribution.version,
        "installed_distribution_matches_reviewed_wheel": True,
        "source_or_vcs_install_rejected": True,
        "wheel_payload_file_count": len(wheel_entries),
        "installed_payload_file_count": len(installed_entries),
        "wheel_payload_file_tree_sha256": canonical_sha256(wheel_entries),
        "installed_payload_file_tree_sha256": canonical_sha256(
            installed_entries
        ),
        "direct_url_sha256": (
            hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest()
            if direct_url_text is not None
            else None
        ),
    }


def verify_wheel_file(
    path: Path,
    *,
    expected_filename: str,
    expected_sha256: str,
) -> dict[str, Any]:
    if path.name != expected_filename:
        raise PermissionError("dependency wheel filename differs")
    if not path.is_file() or path.is_symlink():
        raise PermissionError(
            "dependency wheel must be a regular non-symlink file"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise PermissionError("dependency wheel SHA-256 differs")
    return {
        "path": str(path),
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": actual_sha256,
    }
