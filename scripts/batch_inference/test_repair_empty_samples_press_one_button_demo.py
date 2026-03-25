from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.batch_inference.repair_empty_samples_press_one_button_demo import (
    RepairTask,
    build_backup_path,
    build_repair_tasks,
    run_repair_task,
    scan_empty_sample_records,
)


def _make_args(*, camera_names: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        camera_names=camera_names,
        trajectory_dirname="trajectory",
        external_geom_name="trajectory_valid.h5",
    )


def _write_scene_meta(camera_dir: Path, *, source_frame_indices: list[int], future_len: int) -> None:
    camera_dir.mkdir(parents=True, exist_ok=True)
    (camera_dir / "scene_meta.json").write_text(
        json.dumps(
            {
                "source_frame_indices": list(source_frame_indices),
                "future_len": int(future_len),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_sample(sample_path: Path, *, query_frame: int, valid_count: int) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    traj_valid_mask = np.zeros(4, dtype=bool)
    traj_valid_mask[:valid_count] = True
    np.savez(
        sample_path,
        query_frame_index=np.array([query_frame], dtype=np.int32),
        segment_frame_indices=np.arange(query_frame, query_frame + 3, dtype=np.int32),
        traj_valid_mask=traj_valid_mask,
    )


class RepairEmptySamplesHelperTests(unittest.TestCase):
    def test_scan_empty_sample_records_keeps_only_empty_samples(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            episode_dir = base_path / "episode_00000_blue"
            camera_dir = episode_dir / "trajectory" / "varied_camera_1"
            _write_scene_meta(camera_dir, source_frame_indices=list(range(20)), future_len=32)
            _write_sample(camera_dir / "samples" / "varied_camera_1_10.npz", query_frame=10, valid_count=0)
            _write_sample(camera_dir / "samples" / "varied_camera_1_11.npz", query_frame=11, valid_count=2)

            records = scan_empty_sample_records(
                base_path=base_path,
                args=_make_args(camera_names=["varied_camera_1"]),
                out_root=None,
                episodes=[episode_dir],
            )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].episode_name, "episode_00000_blue")
            self.assertEqual(records[0].camera_name, "varied_camera_1")
            self.assertEqual(records[0].query_frame_local, 10)
            self.assertEqual(records[0].query_frame_source, 10)
            self.assertEqual(records[0].before_valid_count, 0)
            self.assertEqual(records[0].segment_len, 3)

    def test_build_repair_tasks_groups_by_episode_camera_and_writes_schedule(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)
            episode_dir = base_path / "episode_00000_blue"
            camera_dir = episode_dir / "trajectory" / "varied_camera_1"
            _write_scene_meta(camera_dir, source_frame_indices=list(range(30)), future_len=32)
            _write_sample(camera_dir / "samples" / "varied_camera_1_10.npz", query_frame=10, valid_count=0)
            _write_sample(camera_dir / "samples" / "varied_camera_1_14.npz", query_frame=14, valid_count=0)

            records = scan_empty_sample_records(
                base_path=base_path,
                args=_make_args(camera_names=["varied_camera_1"]),
                out_root=None,
                episodes=[episode_dir],
            )
            report_root = base_path / "reports"
            tasks = build_repair_tasks(records, report_root=report_root)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].query_frames_local, (10, 14))
            self.assertEqual(tasks[0].query_frames_source, (10, 14))
            schedule_path = Path(tasks[0].schedule_path)
            payload = json.loads(schedule_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["query_frame_local_indices"], [10, 14])
            self.assertEqual(payload["query_frame_source_indices"], [10, 14])

    def test_build_backup_path_preserves_output_relative_layout(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "dataset"
            sample_path = (
                base_path
                / "episode_00000_blue"
                / "trajectory"
                / "varied_camera_1"
                / "samples"
                / "varied_camera_1_10.npz"
            )
            backup_root = Path(tmpdir) / "backup"
            expected = (
                backup_root
                / "episode_00000_blue"
                / "trajectory"
                / "varied_camera_1"
                / "samples"
                / "varied_camera_1_10.npz"
            )

            actual = build_backup_path(sample_path, mirror_root=base_path, backup_root=backup_root)

            self.assertEqual(actual, expected.resolve())

    def test_run_repair_task_removes_short_tail_sample_after_backup(self):
        with TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "dataset"
            episode_dir = base_path / "episode_00000_blue"
            out_episode_dir = episode_dir / "trajectory"
            sample_path = (
                out_episode_dir
                / "varied_camera_1"
                / "samples"
                / "varied_camera_1_10.npz"
            )
            _write_sample(sample_path, query_frame=10, valid_count=0)
            schedule_path = base_path / "reports" / "schedule.json"
            schedule_path.parent.mkdir(parents=True, exist_ok=True)
            schedule_path.write_text("{}", encoding="utf-8")

            task = RepairTask(
                task_index=1,
                total_tasks=1,
                episode_name=episode_dir.name,
                episode_dir=str(episode_dir.resolve()),
                out_episode_dir=str(out_episode_dir.resolve()),
                camera_name="varied_camera_1",
                query_frames_local=(10,),
                query_frames_source=(10,),
                sample_paths=(str(sample_path.resolve()),),
                schedule_path=str(schedule_path.resolve()),
            )
            backup_root = base_path / "backup"

            fake_args = SimpleNamespace(
                checkpoint="./checkpoints/tapip3d_final.pth",
                depth_pose_method="external",
                grid_size=80,
            )

            with mock.patch(
                "scripts.batch_inference.repair_empty_samples_press_one_button_demo.batch_infer.build_camera_args",
                return_value=SimpleNamespace(
                    depth_pose_method="external",
                    grid_size=80,
                ),
            ), mock.patch.dict(
                "scripts.batch_inference.repair_empty_samples_press_one_button_demo.infer.video_depth_pose_dict",
                {"external": lambda _camera_args: object()},
                clear=False,
            ), mock.patch(
                "scripts.batch_inference.repair_empty_samples_press_one_button_demo.infer.process_single_video",
                return_value={
                    "query_frame_results": {},
                    "query_frame_metadata": {
                        "skipped_short_tail_query_frame_indices_local": [10],
                    },
                },
            ), mock.patch(
                "scripts.batch_inference.repair_empty_samples_press_one_button_demo.batch_infer.safe_empty_cuda_cache"
            ):
                ok, retryable, task_record = run_repair_task(
                    task=task,
                    args=fake_args,
                    model_3dtracker=object(),
                    backup_root=backup_root,
                    mirror_root=base_path,
                )

            backup_path = build_backup_path(sample_path, mirror_root=base_path, backup_root=backup_root)
            self.assertTrue(ok)
            self.assertFalse(retryable)
            self.assertEqual(task_record["removed_short_tail_count"], 1)
            self.assertEqual(task_record["failed_sample_count"], 0)
            self.assertEqual(task_record["sample_results"][0]["status"], "removed_short_tail")
            self.assertEqual(task_record["sample_results"][0]["skip_reason"], "short_tail_segment_len<=8")
            self.assertFalse(sample_path.exists())
            self.assertTrue(backup_path.exists())


if __name__ == "__main__":
    unittest.main()
