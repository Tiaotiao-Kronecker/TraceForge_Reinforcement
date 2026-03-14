import numpy as np
import sys

def verify_traj_valid_mask(npz_path):
    """Verify traj_valid_mask field"""
    data = np.load(npz_path)

    # Check field exists
    assert "traj_valid_mask" in data, "traj_valid_mask field not found"

    mask = data["traj_valid_mask"]
    traj = data["traj"]

    # Check shape
    N, T, D = traj.shape
    assert mask.shape == (N,), f"Shape error: expected ({N},), got {mask.shape}"

    # Check type
    assert mask.dtype == bool, f"Type error: expected bool, got {mask.dtype}"

    # Statistics
    valid_count = mask.sum()
    total_count = len(mask)
    valid_ratio = valid_count / total_count

    print(f"✓ {npz_path}")
    print(f"  Total trajectories: {total_count}")
    print(f"  Valid trajectories: {valid_count} ({valid_ratio*100:.1f}%)")
    print(f"  Invalid trajectories: {total_count - valid_count} ({(1-valid_ratio)*100:.1f}%)")

    # Analyze invalid trajectories
    if valid_count < total_count:
        invalid_traj = traj[~mask]
        print(f"\n  Invalid trajectory analysis:")
        for i, t in enumerate(invalid_traj[:3]):
            valid_frames = np.isfinite(t).all(axis=-1)
            valid_count_i = valid_frames.sum()
            print(f"    Trajectory {i}: valid_frames={valid_count_i}")
            if valid_count_i > 0:
                z_values = t[valid_frames, 2]
                print(f"      Depth range: {z_values.min():.3f}~{z_values.max():.3f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_traj_valid_mask.py <npz_file_path>")
        sys.exit(1)
    verify_traj_valid_mask(sys.argv[1])
