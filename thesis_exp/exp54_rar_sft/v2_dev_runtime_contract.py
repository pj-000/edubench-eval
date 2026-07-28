"""Complete source and dependency identity for formal Exp54 V2 dev."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
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
        directory_info = direct_url.get("dir_info")
        if isinstance(directory_info, dict) and directory_info.get(
            "editable"
        ):
            raise PermissionError(
                f"{distribution_name}: editable install is prohibited"
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
