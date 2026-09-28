"""The directory label keeps model identity but omits redundant data settings."""
import unittest

from stmoe_imputer.utils.run_naming import experiment_label


class ExperimentLabelTests(unittest.TestCase):
    def test_depth_and_mask_name(self):
        self.assertEqual(
            experiment_label("v24_coe_depth_s1_top6_random_rate0.4_seed7"),
            "depth_s1_top6",
        )

    def test_partner_variants(self):
        self.assertEqual(
            experiment_label("v24_coe_partner_native4_residual4_partner_fusion_headonly_mixed9_rate0.4_updates_seed7"),
            "partner_native4_partner_fusion_headonly",
        )

    def test_expert_count_kept_and_epoch_removed(self):
        self.assertEqual(
            experiment_label("v24_pair_s3_e6_top2_random_rate0.4_e30_seed7"),
            "pair_s3_e6_top2",
        )

    def test_dataset_name_is_not_repeated(self):
        self.assertEqual(
            experiment_label("v24_coe_depth_s3_top2_TaxiBJ_random_rate0.4_seed7"),
            "depth_s3_top2",
        )

    def test_other_mask_and_version(self):
        self.assertEqual(
            experiment_label("v24_coe_team_accept_v4_team_e2_original_random_rate0.4_updates_seed7"),
            "team_accept_e2",
        )


if __name__ == "__main__":
    unittest.main()
