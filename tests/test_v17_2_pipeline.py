from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType


def load_pipeline_common() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "v17.2-single"
        / "pipeline_common.py"
    )
    spec = importlib.util.spec_from_file_location("v17_2_pipeline_common", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline_common()


class V172PipelineTest(unittest.TestCase):
    def test_protocol_audit_allows_only_adapter_and_version_metadata(self) -> None:
        common = ("TaxiBJ", "random", "0.4", 42)
        full = pipeline.build_resolved_config(*common, version="full")
        e1 = pipeline.build_resolved_config(*common, version="e1")
        candidate = pipeline.build_resolved_config(*common, version="v17_2")
        report = pipeline.build_protocol_audit(full, e1, candidate)
        self.assertTrue(report["passed"], report["unexpected_differences"])
        self.assertFalse(report["contract"]["e1_adapter_enabled"])
        self.assertFalse(report["contract"]["v17_2_adapter_enabled"])

    def test_protocol_audit_rejects_training_drift(self) -> None:
        common = ("BikeNYC", "fixed", "0.8", 42)
        full = pipeline.build_resolved_config(*common, version="full")
        e1 = pipeline.build_resolved_config(*common, version="e1")
        candidate = pipeline.build_resolved_config(*common, version="v17_2")
        candidate["train"]["lr_main"] = 0.123
        report = pipeline.build_protocol_audit(full, e1, candidate)
        self.assertFalse(report["passed"])
        self.assertIn(
            "train.lr_main",
            {item["path"] for item in report["unexpected_differences"]},
        )


if __name__ == "__main__":
    unittest.main()
