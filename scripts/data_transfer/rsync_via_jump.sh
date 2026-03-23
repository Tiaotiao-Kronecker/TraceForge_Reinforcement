#!/usr/bin/env bash

set -euo pipefail

ORIGINAL_ARGS=("$@")

usage() {
    cat <<'EOF'
Usage:
  scripts/data_transfer/rsync_via_jump.sh \
    --src /local/source_dir/ \
    --dst /remote/target_dir/ \
    --target-host 10.0.0.8 \
    [--jump-host jump.example.com] \
    [--jump-user alice] \
    [--target-user bob] \
    [--jump-port 22] \
    [--target-port 22] \
    [--identity-file ~/.ssh/id_rsa] \
    [--ssh-config-file ~/.ssh/config] \
    [--bwlimit 0] \
    [--delete] \
    [--dry-run] \
    [--print-only] \
    [--tmux-session traceforge-transfer] \
    [--ssh-option StrictHostKeyChecking=no] \
    [--rsync-arg --exclude=.cache]

Description:
  Resumable directory/file transfer to a target machine reached either
  directly or through a single SSH jump host. The script uses:

    rsync -aH --info=progress2 --partial --append-verify

  It also creates the remote destination directory before starting the
  transfer. Use --tmux-session for long-running jobs.

Notes:
  - rsync trailing slash semantics are preserved.
  - If --src ends with '/', rsync copies directory contents.
  - If --src does not end with '/', rsync copies the directory itself.
  - --bwlimit is in KiB/s, matching rsync's convention. Use 0 to disable.
  - --dry-run still contacts the remote host to compare file lists.
  - Use --print-only to print the final ssh/rsync commands without connecting.
  - By default the script uses ssh -F /dev/null for predictable behavior.
  - Pass --ssh-config-file when you explicitly want to use a custom ssh config.
  - If target_host is an alias in your ssh config, you can omit --jump-host and
    rely on ProxyJump/User/Port settings from that alias.
  - Repeat --ssh-option or --rsync-arg as needed.
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

resolve_script_path() {
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "$0" 2>/dev/null && return 0
    fi
    if command -v realpath >/dev/null 2>&1; then
        realpath "$0" 2>/dev/null && return 0
    fi
    printf '%s\n' "$0"
}

SRC=""
DST=""
JUMP_HOST=""
JUMP_USER=""
JUMP_PORT=""
TARGET_HOST=""
TARGET_USER=""
TARGET_PORT=""
IDENTITY_FILE=""
SSH_CONFIG_FILE="/dev/null"
BWLIMIT=""
DRY_RUN="0"
PRINT_ONLY="0"
DELETE_MODE="0"
TMUX_SESSION=""
declare -a SSH_OPTIONS=()
declare -a RSYNC_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)
            [[ $# -ge 2 ]] || die "Missing value for --src"
            SRC="$2"
            shift 2
            ;;
        --dst)
            [[ $# -ge 2 ]] || die "Missing value for --dst"
            DST="$2"
            shift 2
            ;;
        --jump-host)
            [[ $# -ge 2 ]] || die "Missing value for --jump-host"
            JUMP_HOST="$2"
            shift 2
            ;;
        --jump-user)
            [[ $# -ge 2 ]] || die "Missing value for --jump-user"
            JUMP_USER="$2"
            shift 2
            ;;
        --jump-port)
            [[ $# -ge 2 ]] || die "Missing value for --jump-port"
            JUMP_PORT="$2"
            shift 2
            ;;
        --target-host)
            [[ $# -ge 2 ]] || die "Missing value for --target-host"
            TARGET_HOST="$2"
            shift 2
            ;;
        --target-user)
            [[ $# -ge 2 ]] || die "Missing value for --target-user"
            TARGET_USER="$2"
            shift 2
            ;;
        --target-port)
            [[ $# -ge 2 ]] || die "Missing value for --target-port"
            TARGET_PORT="$2"
            shift 2
            ;;
        --identity-file)
            [[ $# -ge 2 ]] || die "Missing value for --identity-file"
            IDENTITY_FILE="$2"
            shift 2
            ;;
        --ssh-config-file)
            [[ $# -ge 2 ]] || die "Missing value for --ssh-config-file"
            SSH_CONFIG_FILE="$2"
            shift 2
            ;;
        --bwlimit)
            [[ $# -ge 2 ]] || die "Missing value for --bwlimit"
            BWLIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --print-only)
            PRINT_ONLY="1"
            shift
            ;;
        --delete)
            DELETE_MODE="1"
            shift
            ;;
        --tmux-session)
            [[ $# -ge 2 ]] || die "Missing value for --tmux-session"
            TMUX_SESSION="$2"
            shift 2
            ;;
        --ssh-option)
            [[ $# -ge 2 ]] || die "Missing value for --ssh-option"
            SSH_OPTIONS+=("$2")
            shift 2
            ;;
        --rsync-arg)
            [[ $# -ge 2 ]] || die "Missing value for --rsync-arg"
            RSYNC_EXTRA_ARGS+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$SRC" ]] || die "--src is required"
[[ -n "$DST" ]] || die "--dst is required"
[[ -n "$TARGET_HOST" ]] || die "--target-host is required"
[[ -e "$SRC" ]] || die "Source path does not exist: $SRC"
if [[ -z "$JUMP_HOST" && ( -n "$JUMP_USER" || -n "$JUMP_PORT" ) ]]; then
    die "--jump-user/--jump-port require --jump-host"
fi

require_cmd ssh
require_cmd rsync

if [[ -n "$TMUX_SESSION" && -z "${TMUX:-}" && "${TRACEFORGE_TRANSFER_IN_TMUX:-0}" != "1" ]]; then
    require_cmd tmux
    SCRIPT_PATH="$(resolve_script_path)"
    TMUX_CMD="$(printf '%q ' env TRACEFORGE_TRANSFER_IN_TMUX=1 "$SCRIPT_PATH" "${ORIGINAL_ARGS[@]}")"
    tmux new-session -d -s "$TMUX_SESSION" "$TMUX_CMD"
    echo "Started tmux session: $TMUX_SESSION"
    echo "Attach with: tmux attach -t $TMUX_SESSION"
    exit 0
fi

TARGET_SPEC="$TARGET_HOST"
if [[ -n "$TARGET_USER" ]]; then
    TARGET_SPEC="${TARGET_USER}@${TARGET_SPEC}"
fi

SSH_CMD=(ssh -F "$SSH_CONFIG_FILE")
if [[ -n "$JUMP_HOST" ]]; then
    JUMP_SPEC="$JUMP_HOST"
    if [[ -n "$JUMP_USER" ]]; then
        JUMP_SPEC="${JUMP_USER}@${JUMP_SPEC}"
    fi
    if [[ -n "$JUMP_PORT" ]]; then
        JUMP_SPEC="${JUMP_SPEC}:${JUMP_PORT}"
    fi
    SSH_CMD+=(-J "$JUMP_SPEC")
fi
if [[ -n "$TARGET_PORT" ]]; then
    SSH_CMD+=(-p "$TARGET_PORT")
fi
if [[ -n "$IDENTITY_FILE" ]]; then
    SSH_CMD+=(-i "$IDENTITY_FILE")
fi
for opt in "${SSH_OPTIONS[@]}"; do
    SSH_CMD+=(-o "$opt")
done

REMOTE_MKDIR_CMD="mkdir -p -- $(printf '%q' "$DST")"

echo "Preparing remote directory: ${TARGET_SPEC}:${DST}"
printf '  %q' "${SSH_CMD[@]}"
printf ' %q %q\n' "$TARGET_SPEC" "$REMOTE_MKDIR_CMD"
if [[ "$PRINT_ONLY" != "1" && "$DRY_RUN" != "1" ]]; then
    "${SSH_CMD[@]}" "$TARGET_SPEC" "$REMOTE_MKDIR_CMD"
fi

SSH_TRANSPORT="$(printf '%q ' "${SSH_CMD[@]}")"
SSH_TRANSPORT="${SSH_TRANSPORT% }"

RSYNC_CMD=(
    rsync
    -aH
    -s
    --info=progress2
    --partial
    --append-verify
)

if [[ "$DRY_RUN" == "1" ]]; then
    RSYNC_CMD+=(-n)
fi

if [[ "$DELETE_MODE" == "1" ]]; then
    RSYNC_CMD+=(--delete)
fi

if [[ -n "$BWLIMIT" && "$BWLIMIT" != "0" ]]; then
    RSYNC_CMD+=("--bwlimit=$BWLIMIT")
fi

for arg in "${RSYNC_EXTRA_ARGS[@]}"; do
    RSYNC_CMD+=("$arg")
done

RSYNC_CMD+=(
    -e "$SSH_TRANSPORT"
    "$SRC"
    "${TARGET_SPEC}:${DST}"
)

echo "Running transfer:"
printf '  %q' "${RSYNC_CMD[@]}"
printf '\n'

if [[ "$PRINT_ONLY" != "1" ]]; then
    "${RSYNC_CMD[@]}"
fi
