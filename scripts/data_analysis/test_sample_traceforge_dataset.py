import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_analysis.sample_traceforge_dataset import (
    build_sample_manifest,
    get_all_cases,
    is_valid_press_one_button_episode,
    materialize_sampled_cases,
    parse_exclude_dir_names,
    sample_random,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _build_press_one_button_episode(root: Path, episode_name: str) -> Path:
    episode_dir = root / episode_name
    _touch(episode_dir / "trajectory_valid.h5")
    _touch(episode_dir / "rgb" / "varied_camera_1" / "000000.png")
    _touch(episode_dir / "depth" / "varied_camera_1" / "000000.npy")
    _touch(episode_dir / "rgb" / "varied_camera_3" / "000000.png")
    _touch(episode_dir / "depth" / "varied_camera_3" / "000000.npy")
    _touch(episode_dir / "lang.txt")
    _touch(episode_dir / "trajectory" / "old" / "sample.npz")
    return episode_dir


class PressOneButtonSamplingTests(unittest.TestCase):
    def test_is_valid_press_one_button_episode_accepts_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = _build_press_one_button_episode(Path(tmpdir), "episode_00001_green")
            self.assertTrue(is_valid_press_one_button_episode(episode_dir))

    def test_get_all_cases_filters_press_one_button_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _build_press_one_button_episode(root, "episode_00001_green")
            (root / "notes").mkdir()
            cases = get_all_cases(root, valid_only=True, layout="press_one_button")
            self.assertEqual(cases, ["episode_00001_green"])

    def test_sample_random_is_deterministic_with_seed(self):
        cases = [f"episode_{idx:05d}_green" for idx in range(10)]
        sampled_a = sample_random(cases, 4, seed=42)
        sampled_b = sample_random(cases, 4, seed=42)
        self.assertEqual(sampled_a, sampled_b)

    def test_materialize_sampled_cases_copy_excludes_trajectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_root = root / "source"
            dst_root = root / "sampled"
            _build_press_one_button_episode(src_root, "episode_00001_green")

            materialize_sampled_cases(
                data_dir=src_root,
                sampled=["episode_00001_green"],
                output_dir=dst_root,
                mode="copy",
                exclude_dir_names=["trajectory"],
            )

            copied = dst_root / "episode_00001_green"
            self.assertTrue((copied / "rgb" / "varied_camera_1" / "000000.png").is_file())
            self.assertFalse((copied / "trajectory").exists())

    def test_materialize_sampled_cases_raises_when_destination_exists_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_root = root / "source"
            dst_root = root / "sampled"
            _build_press_one_button_episode(src_root, "episode_00001_green")
            (dst_root / "episode_00001_green").mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                materialize_sampled_cases(
                    data_dir=src_root,
                    sampled=["episode_00001_green"],
                    output_dir=dst_root,
                    mode="copy",
                    exclude_dir_names=["trajectory"],
                )

    def test_build_sample_manifest_uses_sampled_output_as_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "source"
            sampled_root = root / "sampled"
            manifest = build_sample_manifest(
                data_dir=source_root,
                sampled=["episode_00001_green", "episode_00002_blue"],
                layout="press_one_button",
                method="random",
                n_requested=30,
                seed=42,
                exclude_dir_names=["trajectory"],
                output_dir=sampled_root,
            )

            self.assertEqual(manifest["dataset_root"], str(sampled_root.resolve()))
            self.assertEqual(manifest["source_dataset_root"], str(source_root.resolve()))
            self.assertEqual(manifest["episodes"], ["episode_00001_green", "episode_00002_blue"])
            self.assertEqual(manifest["excluded_dir_names"], ["trajectory"])

    def test_parse_exclude_dir_names_skips_empty_entries(self):
        self.assertEqual(parse_exclude_dir_names("trajectory,,mask"), ["trajectory", "mask"])


if __name__ == "__main__":
    unittest.main()
