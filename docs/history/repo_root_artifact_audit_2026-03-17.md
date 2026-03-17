# Repo Root Artifact Audit (2026-03-17)

This note audits repository-root artifacts that fell outside the earlier
directory-focused cleanup pass.

If an item here conflicts with the maintained docs, prefer `docs/README.md`.

## Summary

| item | status | size | current references | recommended action | delete blocked by user review |
|------|--------|------|--------------------|--------------------|-------------------------------|
| `00001/` | tracked | `14M` | no explicit path-level references; only generic numeric-case examples | deleted by user decision | resolved |
| `rerun_failed_4.log` | ignored | `262 lines` | none | deleted during cleanup | resolved |
| `sampled_1000.txt` | ignored | `999 lines` | only appears as a generic example filename in docs/scripts | deleted during cleanup | resolved |
| `discussion.txt` | ignored | local note | none | deleted during cleanup | resolved |
| `failed_traj_ids.txt` | ignored | local list | none | deleted during cleanup | resolved |
| `inference_extdepth_viz.log` | ignored | local log | none | deleted during cleanup | resolved |
| `inference_external_geom.log` | ignored | local log | none | deleted during cleanup | resolved |

## `00001/`

Status:

- git-tracked directory
- top-level Bridge/legacy-style sample case

Observed structure:

- `images0/`, `images1/`, `images2/`
- `depth_images0/`
- `agent_data.pkl`
- `obs_dict.pkl`
- `policy_out.pkl`
- `lang.txt`

Current references:

- No maintained doc or script hardcodes the path `00001/`
- Some scripts mention generic numeric case IDs such as `00000`, `00001`,
  `01625` to describe Bridge-style directory formats

Assessment:

- The directory is not part of the current documented public interface
- It may still be useful as a historical sample, but there is no hard evidence
  that current tests or docs require it

Resolution:

- deleted after explicit user confirmation
- no follow-up action remains

## `rerun_failed_4.log`

Status:

- ignored by `.gitignore` via `*.log`
- repository-root rerun log

Assessment:

- one-off local debugging output
- not referenced by maintained docs or scripts

Recommended action:

- deleted during cleanup

## `sampled_1000.txt`

Status:

- ignored by `.gitignore` via `sampled_*.txt`
- repository-root sampled case list

Assessment:

- one-off local sampling output
- the exact filename is only used as a generic example in docs and script
  docstrings, not as a required repository asset

Recommended action:

- deleted during cleanup

## Additional ignored root files removed

The following ignored repository-root files were also removed in the same
cleanup pass:

- `discussion.txt`
- `failed_traj_ids.txt`
- `inference_extdepth_viz.log`
- `inference_external_geom.log`
