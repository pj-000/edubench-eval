from __future__ import annotations

import unittest

from thesis_exp.exp54_rar_sft.audit_sft_dev_selection_freeze import (
    independent_backend_contract,
)
from thesis_exp.exp54_rar_sft.freeze_sft_dev_selection import (
    inference_backend_contract,
)


def backend(*, forced_completion_count: int = 7) -> dict[str, object]:
    return {
        "budget_boundary_policy": (
            "truncate_rationale_prefix_and_force_json_close"
        ),
        "compact_json_whitespace_disabled": True,
        "cuda": "11.8",
        "forced_completion_count": forced_completion_count,
        "name": "vllm",
        "torch": "2.7.1+cu118",
        "version": "0.10.0",
        "xgrammar_source_sha256": "a" * 64,
    }


class SFTDevSelectionFreezeTest(unittest.TestCase):
    def test_run_specific_forced_count_is_not_a_protocol_difference(
        self,
    ) -> None:
        self.assertEqual(
            inference_backend_contract(
                backend(forced_completion_count=0)
            ),
            inference_backend_contract(
                backend(forced_completion_count=194)
            ),
        )

    def test_production_and_independent_contracts_agree(self) -> None:
        value = backend()
        self.assertEqual(
            inference_backend_contract(value),
            independent_backend_contract(value),
        )

    def test_backend_contract_rejects_missing_field(self) -> None:
        value = backend()
        del value["xgrammar_source_sha256"]
        with self.assertRaises(ValueError):
            inference_backend_contract(value)

    def test_backend_contract_rejects_extra_field(self) -> None:
        value = backend()
        value["unreviewed"] = True
        with self.assertRaises(ValueError):
            independent_backend_contract(value)

    def test_backend_contract_rejects_non_integer_forced_count(self) -> None:
        value = backend()
        value["forced_completion_count"] = "7"
        with self.assertRaises(ValueError):
            inference_backend_contract(value)

    def test_real_backend_change_remains_visible(self) -> None:
        changed = backend()
        changed["version"] = "0.11.0"
        self.assertNotEqual(
            inference_backend_contract(backend()),
            inference_backend_contract(changed),
        )


if __name__ == "__main__":
    unittest.main()
