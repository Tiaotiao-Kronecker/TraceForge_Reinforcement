import unittest

from scripts.data_analysis.benchmark_num_iters_sweep import (
    build_variant_specs,
    format_num_iters_variant_name,
    parse_num_iters_values,
)


class NumItersVariantSpecTests(unittest.TestCase):
    def test_parse_num_iters_values_requires_positive_ints(self):
        with self.assertRaises(ValueError):
            parse_num_iters_values("6,0,4")

    def test_parse_num_iters_values_deduplicates_in_order(self):
        self.assertEqual(parse_num_iters_values("6,5,6,4,5"), [6, 5, 4])

    def test_build_variant_specs_orders_baseline_first(self):
        specs = build_variant_specs(
            num_iters_values=[6, 5, 4],
            baseline_num_iters=6,
            support_grid_ratio=0.0,
        )

        self.assertEqual([spec["name"] for spec in specs], ["iters_6", "iters_5", "iters_4"])
        self.assertEqual([spec["num_iters"] for spec in specs], [6, 5, 4])
        self.assertEqual([bool(spec["is_baseline"]) for spec in specs], [True, False, False])
        self.assertTrue(all(spec["support_grid_ratio"] == 0.0 for spec in specs))

    def test_build_variant_specs_requires_baseline_in_values(self):
        with self.assertRaises(ValueError):
            build_variant_specs(
                num_iters_values=[5, 4],
                baseline_num_iters=6,
                support_grid_ratio=0.0,
            )

    def test_build_variant_specs_supports_subset_selection(self):
        specs = build_variant_specs(
            num_iters_values=[6, 5, 4],
            baseline_num_iters=6,
            support_grid_ratio=0.0,
            selected_variant_names=[format_num_iters_variant_name(6), format_num_iters_variant_name(4)],
        )

        self.assertEqual([spec["name"] for spec in specs], ["iters_6", "iters_4"])

    def test_build_variant_specs_subset_must_include_baseline(self):
        with self.assertRaises(ValueError):
            build_variant_specs(
                num_iters_values=[6, 5, 4],
                baseline_num_iters=6,
                support_grid_ratio=0.0,
                selected_variant_names=[format_num_iters_variant_name(5)],
            )


if __name__ == "__main__":
    unittest.main()
