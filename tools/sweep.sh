#!/usr/bin/env bash
# =============================================================================
#  llmbench quantization sweep — automates: start llama-server -> run bench -> stop
#
#  For each quant, start llama-server, run the enabled suites in order
#  (l6 / l7 / culture / unc / sec / secaug / gen / med / write), shut the
#  server down, and move on to the next quant.
#
#  Usage:  cp tools/sweep.conf.example tools/sweep.conf   # put your real paths here (not git-tracked)
#          tools/sweep.sh --list         # only show what would run
#          tools/sweep.sh --dry-run      # only show the commands
#          tools/sweep.sh                # production run
#  Settings are overridden in this order: block below / tools/sweep.conf / env vars / CLI.
#  See docs/SWEEP.md for details.
# =============================================================================

# =============================================================================
#  ◆ Configuration — this is the only section you need to edit ◆
#    Everything uses the ${VAR:-default} form. Env vars can override temporarily:
#      RUNS_L7=5 RUN_CULTURE=0 ./sweep.sh
# =============================================================================

# ── Paths ──────────────────────────────────────────────────────────────────
#   REPO_DIR defaults to "the directory one level above this script".
#   As long as this file lives at tools/sweep.sh, no edits are needed after a clone.
_SWEEP_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$_SWEEP_SELF_DIR")}"   # llmbench repository
#   ↓ When left empty, derive from REPO_DIR (follows a REPO_DIR change made in a config file).
#     Only set a value when one path must point somewhere else.
LLMBENCH="${LLMBENCH:-}"        # default: $REPO_DIR/.venv/bin/llmbench
CONFIG="${CONFIG:-}"            # default: $REPO_DIR/config.yaml
TASKS_DIR="${TASKS_DIR:-}"      # default: $REPO_DIR/tasks
OUT_ROOT="${OUT_ROOT:-}"        # default: $REPO_DIR/_OUTPUTS/sweep  (logs/state/summary)
RESULTS_DIR="${RESULTS_DIR:-}"  # default: $REPO_DIR/results         (llmbench --output)
MODEL_KEY="${MODEL_KEY:-local-openai}"                  # models: key in config.yaml
MODEL_DIR="${MODEL_DIR:-/llm/models/Qwen3.8-27B-GGUF}"  # where the gguf files live
MODEL_PREFIX="${MODEL_PREFIX:-Qwen3.8-27B}"             # <PREFIX>-<QUANT>.gguf
LABEL_PREFIX="${LABEL_PREFIX:-}"                        # prefix for result labels/state
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"            # path to llama-server

# ── Target quants ──────────────────────────────────────────────────────────
#   Explicit list … "Q4_K_M Q6_K Q8_0"  (space- or comma-separated)
#   Auto-detect   … "auto"  → every *.gguf in MODEL_DIR (sorted by filename)
QUANTS="${QUANTS:-Q4_K_M Q6_K Q8_0}"
QUANT_EXCLUDE="${QUANT_EXCLUDE:-}"      # regex to exclude in auto mode (e.g. "UD-|Q8_0")

# ── Suites: on/off, runs, args ─────────────────────────────────────────────
#   RUN_* = 1 runs the suite, 0 skips it. ARGS_* are the ledger flags passed to llmbench run.
RUN_L6="${RUN_L6:-1}"                                   # default 40 tasks + 20 L6(architect)
RUNS_L6="${RUNS_L6:-5}"
ARGS_L6="${ARGS_L6:---with-l6}"

RUN_L7="${RUN_L7:-1}"                                   # L7(grandmaster) 16 tasks only
RUNS_L7="${RUNS_L7:-3}"
ARGS_L7="${ARGS_L7:---only-l7}"

RUN_CULTURE="${RUN_CULTURE:-1}"                         # Japanese internet-meme knowledge only
RUNS_CULTURE="${RUNS_CULTURE:-3}"
ARGS_CULTURE="${ARGS_CULTURE:---only-culture --lang ja}"

RUN_UNC="${RUN_UNC:-1}"                                 # over-refusal check only
RUNS_UNC="${RUNS_UNC:-3}"
ARGS_UNC="${ARGS_UNC:---only-unc}"

# ── Domain suites (all disabled by default; enable with RUN_*=1) ───────────
RUN_SEC="${RUN_SEC:-0}"                                 # security detection, 17 tasks
RUNS_SEC="${RUNS_SEC:-3}"
ARGS_SEC="${ARGS_SEC:---only-sec}"

RUN_SECAUG="${RUN_SECAUG:-0}"                           # security perturbed variants, 34 tasks
RUNS_SECAUG="${RUNS_SECAUG:-3}"
ARGS_SECAUG="${ARGS_SECAUG:---only-secaug}"

RUN_GEN="${RUN_GEN:-0}"                                 # general instruction following
RUNS_GEN="${RUNS_GEN:-3}"
ARGS_GEN="${ARGS_GEN:---only-gen}"

RUN_MED="${RUN_MED:-0}"                                 # medical QA (reference values)
RUNS_MED="${RUNS_MED:-3}"
ARGS_MED="${ARGS_MED:---only-med --lang ja}"

RUN_WRITE="${RUN_WRITE:-0}"                             # creative writing (judge-scored, experimental)
RUNS_WRITE="${RUNS_WRITE:-1}"
ARGS_WRITE="${ARGS_WRITE:---only-write}"

SUITE_ORDER="${SUITE_ORDER:-l6 l7 culture unc sec secaug gen med write}"  # run order (RUN_*=0 are skipped)
#   Suites not listed in SUITE_ORDER but with RUN_*=1 are warned about and appended at the end.

# ── Per-suite thinking limit (reasoning_max_tokens) ────────────────────────
#   REASONING_MAX_TOKENS_<SUITE>=N runs that suite with a temp config whose
#   models.<MODEL_KEY>.reasoning_max_tokens is rewritten to N.
#   When empty, the value in config.yaml is kept.
#   Measured (OrcaSAQ-2-27B / sec): 16384 cut off s14's thinking → 32768 passed.
#   ⚠️ A value above max_tokens (after the effective-ctx lowering) has no effect, so warn.
# REASONING_MAX_TOKENS_SEC=32768
# REASONING_MAX_TOKENS_SECAUG=32768

# ── Post-run processing ────────────────────────────────────────────────────
POST_SEC_CLASSIFY="${POST_SEC_CLASSIFY:-1}"  # 1 = after sec/secaug finish, run
                                             # tools/sec_classify.py and save the
                                             # output to logs/<RUN_ID>/<quant>_<suite>_classify.txt

# ── llama-server common args ───────────────────────────────────────────────
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8085}"
NGL="${NGL:-99}"                        # -ngl
FLASH_ATTN="${FLASH_ATTN:-on}"          # -fa (omitted when empty)
CTX="${CTX:-65536}"                     # --ctx-size
PARALLEL="${PARALLEL:-1}"               # --parallel (keep in sync with llmbench --concurrency)
BATCH="${BATCH:-2048}"                  # --batch-size
UBATCH="${UBATCH:-512}"                 # --ubatch-size
KV_TYPE="${KV_TYPE:-}"                  # q8_0 etc → -ctk/-ctv (requires -fa on)
DEVICE="${DEVICE:-}"                    # --device (e.g. CUDA0)
TENSOR_SPLIT="${TENSOR_SPLIT:-}"        # --tensor-split (e.g. 0.5,0.5)
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"   # extra args to append
CONCURRENCY="${CONCURRENCY:-}"          # llmbench --concurrency (empty = not passed)

# ── Per-quant overrides ────────────────────────────────────────────────────
#   Variable names are OVERRIDE_<quant>. Replace the symbols . and - with _.
#   Args written here come **after** the common args, so on a duplicate the later one wins.
OVERRIDE_Q4_K_M="${OVERRIDE_Q4_K_M:-}"
OVERRIDE_Q5_K_M="${OVERRIDE_Q5_K_M:-}"
OVERRIDE_Q6_K="${OVERRIDE_Q6_K:-}"       # e.g. "--ctx-size 32768 -ctk q8_0 -ctv q8_0"
OVERRIDE_Q8_0="${OVERRIDE_Q8_0:-}"
OVERRIDE_UD_Q4_K_XL="${OVERRIDE_UD_Q4_K_XL:-}"
OVERRIDE_UD_Q5_K_XL="${OVERRIDE_UD_Q5_K_XL:-}"
OVERRIDE_UD_Q6_K_XL="${OVERRIDE_UD_Q6_K_XL:-}"

# ── Behavior ───────────────────────────────────────────────────────────────
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"  # max seconds to wait for /health to pass
STOP_WAIT="${STOP_WAIT:-60}"             # seconds to wait after SIGTERM; SIGKILL beyond that
VRAM_SETTLE="${VRAM_SETTLE:-10}"         # pause before starting the next model (seconds)
SUITE_TIMEOUT="${SUITE_TIMEOUT:-0}"      # cap for one suite (seconds). 0 = unlimited
STRIP_SEED_FOR_MULTIRUN="${STRIP_SEED_FOR_MULTIRUN:-1}"
                                         # when runs>1, use a temp config with the
                                         # seed line disabled (the original config.yaml
                                         # is left untouched)
ADJUST_MAX_TOKENS="${ADJUST_MAX_TOKENS:-1}"
                                         # 1 = lower max_tokens in the config to the
                                         # effective ctx. max_tokens >= n_ctx makes
                                         # preflight FAIL (the effective cap becomes
                                         # n_ctx − prompt length, so it has no effect)
MAX_TOKENS="${MAX_TOKENS:-}"             # empty = auto (3/4 of the effective ctx, capped by CAP below)
MAX_TOKENS_CAP="${MAX_TOKENS_CAP:-49152}"  # measured value: even thinking models need no more
                                         # ※ only rewrite when lowering (raising ctx
                                         #   never raises max_tokens on its own)
MTP_AUTO="${MTP_AUTO:-1}"                # 1 = decide MTP presence from the gguf and
                                         # auto-remove --spec-type draft-mtp for files
                                         # without MTP (with it kept, llama-server fails
                                         #  to start with "model doesn't contain MTP
                                         #  layers"). 0 = pass through as before
MTP_DETECT="${MTP_DETECT:-auto}"         # auto = decide with gguf_probe.py when it works,
                                         # otherwise grep the header.
                                         # grep = always grep (no gguf package needed)
MTP_SCAN_BYTES="${MTP_SCAN_BYTES:-33554432}"
                                         # first byte count to read for MTP detection (default 32MiB).
                                         # Tensor names live in the gguf header, so this is usually enough
MTP_SCAN_MAX_BYTES="${MTP_SCAN_MAX_BYTES:-536870912}"
                                         # range to read when not found above (default 512MiB).
                                         # Models with a large vocabulary can have headers over
                                         # 32MiB, so check this far before concluding "absent"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"    # 1 = skip llmbench's preflight
RESUME="${RESUME:-1}"                    # 1 = skip completed (quant,suite) pairs
DRY_RUN="${DRY_RUN:-0}"                  # 1 = only print commands
FORCE_PORT="${FORCE_PORT:-0}"            # 1 = kill an existing process holding PORT
KEEP_GOING="${KEEP_GOING:-1}"            # 1 = move to the next quant even after a failure
SWEEP_CONF="${SWEEP_CONF:-}"             # extra config file (plain VAR=value overrides)

# =============================================================================
#  ◆ End of configuration. Nothing below needs editing ◆
# =============================================================================

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"

# --- Config file ------------------------------------------------------------
#   Priority: explicit -c > env var SWEEP_CONF > tools/sweep.conf > <repo>/sweep.conf
#   --no-conf stops auto-loading, so the script's own defaults are the only source.
#   The conf overrides the block above with plain assignments (VAR=value). CLI options win.
_argv=("$@")
_conf_explicit=""
_no_conf=0
for ((_i = 0; _i < ${#_argv[@]}; _i++)); do
  case "${_argv[$_i]}" in
    -c|--conf)  _conf_explicit="${_argv[$((_i + 1))]:-}" ;;
    --conf=*)   _conf_explicit="${_argv[$_i]#--conf=}" ;;
    --no-conf)  _no_conf=1 ;;
  esac
done

SWEEP_CONF_LOADED=""
if [[ "$_no_conf" != "1" ]]; then
  if [[ -n "$_conf_explicit" ]]; then
    [[ -f "$_conf_explicit" ]] || { echo "Config file not found: $_conf_explicit" >&2; exit 1; }
    SWEEP_CONF="$_conf_explicit"
  elif [[ -n "$SWEEP_CONF" ]]; then
    [[ -f "$SWEEP_CONF" ]] || { echo "Config file not found: $SWEEP_CONF" >&2; exit 1; }
  else
    for _c in "$_SWEEP_SELF_DIR/sweep.conf" "$REPO_DIR/sweep.conf"; do
      if [[ -f "$_c" ]]; then SWEEP_CONF="$_c"; break; fi
    done
  fi
  if [[ -n "$SWEEP_CONF" ]]; then
    # shellcheck disable=SC1090
    source "$SWEEP_CONF"
    SWEEP_CONF_LOADED="$SWEEP_CONF"
  fi
fi

# Paths derived from REPO_DIR are finalized **after** reading the conf.
# (So that editing only REPO_DIR in the conf does not leave the other paths on the old REPO_DIR)
: "${LLMBENCH:=$REPO_DIR/.venv/bin/llmbench}"
: "${CONFIG:=$REPO_DIR/config.yaml}"
: "${TASKS_DIR:=$REPO_DIR/tasks}"
: "${OUT_ROOT:=$REPO_DIR/_OUTPUTS/sweep}"
: "${RESULTS_DIR:=$REPO_DIR/results}"

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

  -c, --conf FILE      use the given config file (plain VAR=value overrides)
      --no-conf        do not read any config file (script defaults only)
      --quants LIST    target quants (comma/space separated, "auto" allowed)
      --suites LIST    suites to run (e.g. l7,unc); everything else is not run
      --skip LIST      drop only the listed suites (e.g. l6)
      --runs N         override --runs of every suite with N
      --ctx N          override --ctx-size with N
      --port N         llama-server port
      --model-dir DIR  where the gguf files live
      --results DIR    output dir for llmbench --output
      --timeout SEC    per-suite cap in seconds (0 = unlimited)
      --skip-preflight skip llmbench's preflight
      --no-resume      re-run even completed work
      --resume         skip completed (quant,suite) pairs (default)
      --force-port     kill the existing process holding the port before starting
      --stop-on-error  abort on the first failure (default: move to the next quant)
  -n, --dry-run        print the commands without running them
  -l, --list           show the targets and exit
  -h, --help           show this help

When -c is omitted, the config file is searched in this order and the first
file found is read:
  1. env var SWEEP_CONF
  2. <directory of this script>/sweep.conf   (= tools/sweep.conf)
  3. <repository>/sweep.conf

Examples:
  cp tools/sweep.conf.example tools/sweep.conf   # put your real paths here (not git-tracked)
  $SCRIPT_NAME --list
  $SCRIPT_NAME --quants Q4_K_M,Q6_K --suites l7,unc
  RUNS_L7=5 RUN_CULTURE=0 $SCRIPT_NAME

To compare two models on the same quant (e.g. an MTP build vs a non-MTP
build), use separate conf files and set LABEL_PREFIX. Result labels become
<prefix>-<quant>-<suite> and the resume state is kept separately in
sweep_state_<prefix>.tsv, so a full run of one is never misread as
"already completed" for the other:
  $SCRIPT_NAME -c tools/sweep_mtp.conf     # LABEL_PREFIX=mtp   → mtp-Q4_K_M-l6
  $SCRIPT_NAME -c tools/sweep_nomtp.conf   # LABEL_PREFIX=nomtp → nomtp-Q4_K_M-l6
EOF
}

# --- CLI (wins over the config) ----------------------------------------------
CLI_SUITES=""
CLI_SKIP=""
CLI_RUNS=""
DO_LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--conf)        shift 2 ;;
    --conf=*)         shift ;;
    --no-conf)        shift ;;
    --quants)         QUANTS="$2"; shift 2 ;;
    --quants=*)       QUANTS="${1#--quants=}"; shift ;;
    --suites)         CLI_SUITES="$2"; shift 2 ;;
    --suites=*)       CLI_SUITES="${1#--suites=}"; shift ;;
    --skip)           CLI_SKIP="$2"; shift 2 ;;
    --skip=*)         CLI_SKIP="${1#--skip=}"; shift ;;
    --runs)           CLI_RUNS="$2"; shift 2 ;;
    --runs=*)         CLI_RUNS="${1#--runs=}"; shift ;;
    --ctx)            CTX="$2"; shift 2 ;;
    --ctx=*)          CTX="${1#--ctx=}"; shift ;;
    --port)           PORT="$2"; shift 2 ;;
    --port=*)         PORT="${1#--port=}"; shift ;;
    --model-dir)      MODEL_DIR="$2"; shift 2 ;;
    --model-dir=*)    MODEL_DIR="${1#--model-dir=}"; shift ;;
    --results)        RESULTS_DIR="$2"; shift 2 ;;
    --results=*)      RESULTS_DIR="${1#--results=}"; shift ;;
    --timeout)        SUITE_TIMEOUT="$2"; shift 2 ;;
    --timeout=*)      SUITE_TIMEOUT="${1#--timeout=}"; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --no-resume)      RESUME=0; shift ;;
    --resume)         RESUME=1; shift ;;
    --force-port)     FORCE_PORT=1; shift ;;
    --stop-on-error)  KEEP_GOING=0; shift ;;
    -n|--dry-run)     DRY_RUN=1; shift ;;
    -l|--list)        DO_LIST=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# =============================================================================
#  Utilities
# =============================================================================
C_RESET=$'\033[0m'; C_B=$'\033[1m'; C_R=$'\033[31m'; C_G=$'\033[32m'; C_Y=$'\033[33m'
[[ -t 1 ]] || { C_RESET=""; C_B=""; C_R=""; C_G=""; C_Y=""; }

ts()   { date "+%Y-%m-%d %H:%M:%S"; }
log()  { printf '%s [sweep] %s\n' "$(ts)" "$*"; }

info() { printf '%s [sweep] %s%s%s\n' "$(ts)" "$C_B" "$*" "$C_RESET"; }
warn() { printf '%s [sweep] %s%s%s\n' "$(ts)" "$C_Y" "$*" "$C_RESET" >&2; }
err()  { printf '%s [sweep] %s%s%s\n' "$(ts)" "$C_R" "$*" "$C_RESET" >&2; }
die()  { err "$*"; exit 1; }

# quant name / suite name → a form usable as a variable name
varname() { printf '%s' "$1" | tr '.-' '__' | tr '[:lower:]' '[:upper:]'; }
# indirect reference (returns an empty string even when undefined)
deref()   { local n="$1"; printf '%s' "${!n:-}"; }
# seconds → h:mm:ss
hms() { local s=$1; printf '%d:%02d:%02d' $((s / 3600)) $((s % 3600 / 60)) $((s % 60)); }

# =============================================================================
#  Resolving the targets
# =============================================================================
list_to_words() { printf '%s' "$1" | tr ',' ' ' | xargs -n1 2>/dev/null || true; }

resolve_quants() {
  local q out=()
  if [[ "${QUANTS,,}" == "auto" ]]; then
    [[ -d "$MODEL_DIR" ]] || die "MODEL_DIR does not exist: $MODEL_DIR"
    local f base name
    while IFS= read -r f; do
      base="$(basename "$f")"
      name="${base%.gguf}"
      # for split gguf, keep only the first shard (-00001-of-0000N)
      if [[ "$name" =~ -[0-9]{5}-of-[0-9]{5}$ ]]; then
        [[ "$name" =~ -00001-of-[0-9]{5}$ ]] || continue
        name="${name%-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]}"
      fi
      name="${name#"$MODEL_PREFIX"-}"
      if [[ -n "$QUANT_EXCLUDE" ]] && printf '%s' "$name" | grep -Eq "$QUANT_EXCLUDE"; then
        continue
      fi
      out+=("$name")
    done < <(find "$MODEL_DIR" -maxdepth 1 -name '*.gguf' | sort)
  else
    for q in $(list_to_words "$QUANTS"); do out+=("$q"); done
  fi
  printf '%s\n' "${out[@]:-}"
}

# return the real gguf file path (first shard for split files)
gguf_path() {
  local q="$1" p
  p="$MODEL_DIR/${MODEL_PREFIX}-${q}.gguf"
  if [[ -f "$p" ]]; then printf '%s' "$p"; return 0; fi
  p="$MODEL_DIR/${q}.gguf"
  if [[ -f "$p" ]]; then printf '%s' "$p"; return 0; fi
  p="$(find "$MODEL_DIR" -maxdepth 1 -name "*${q}*-00001-of-*.gguf" 2>/dev/null | sort | head -1)"
  if [[ -n "$p" ]]; then printf '%s' "$p"; return 0; fi
  p="$(find "$MODEL_DIR" -maxdepth 1 -name "*${q}*.gguf" 2>/dev/null | sort | head -1)"
  if [[ -n "$p" ]]; then printf '%s' "$p"; return 0; fi
  return 1
}

KNOWN_SUITES="l6 l7 culture unc sec secaug gen med write"

# Return SUITE_ORDER + (suites not in SUITE_ORDER but enabled/specified).
# So that RUN_SEC=1 or --suites sec is not silently ignored even when an old
# sweep.conf still has SUITE_ORDER="l6 l7 culture unc".
suite_order_effective() {
  local s order extra=""
  order="$(list_to_words "$SUITE_ORDER")"
  for s in $KNOWN_SUITES $(list_to_words "$CLI_SUITES"); do
    printf '%s\n' $order $extra | grep -qx "$s" && continue
    if [[ "$(deref "RUN_$(varname "$s")")" == "1" ]] \
       || printf '%s\n' $(list_to_words "$CLI_SUITES") | grep -qx "$s"; then
      extra="$extra $s"
      warn "Suite '$s' is not in SUITE_ORDER but is enabled; appending it to the end (SUITE_ORDER=\"$SUITE_ORDER\")"
    fi
  done
  printf '%s\n' $order $extra
}

resolve_suites() {
  local s u want=() sel=""
  [[ -n "$CLI_SUITES" ]] && sel="$(list_to_words "$CLI_SUITES")"
  for s in $(suite_order_effective); do
    u="$(varname "$s")"
    if [[ -n "$sel" ]]; then
      # when --suites is given, take only the suites listed there
      printf '%s\n' $sel | grep -qx "$s" || continue
    else
      [[ "$(deref "RUN_$u")" == "1" ]] || continue
    fi
    if [[ -n "$CLI_SKIP" ]] && printf '%s\n' $(list_to_words "$CLI_SKIP") | grep -qx "$s"; then
      continue
    fi
    [[ -n "$(deref "ARGS_$(varname "$s")")" ]] || die "unknown suite '$s' (ARGS_$(varname "$s") is not defined)"
    want+=("$s")
  done
  printf '%s\n' "${want[@]:-}"
}

suite_runs() {
  local u; u="$(varname "$1")"
  if [[ -n "$CLI_RUNS" ]]; then printf '%s' "$CLI_RUNS"; return; fi
  printf '%s' "$(deref "RUNS_$u")"
}
suite_args() { local u; u="$(varname "$1")"; printf '%s' "$(deref "ARGS_$u")"; }

# =============================================================================
#  llama-server control
# =============================================================================
SERVER_PID=""
SERVER_LOG=""
CURRENT_GGUF=""

# Always pass --noproxy to curl. When http_proxy is set, localhost gets
# swallowed by the proxy and /health never passes (hit in practice)
CURL=(curl -s --noproxy '*')

health_code() {
  local c
  c="$("${CURL[@]}" -o /dev/null -m 5 -w '%{http_code}' "http://$HOST:$PORT/health" 2>/dev/null)" || true
  # on a connection failure curl prints 000 to -w; normalize defensively here
  [[ "$c" =~ ^[0-9]{3}$ ]] || c="000"
  printf '%s' "$c"
}

port_busy() { [[ "$(health_code)" != "000" ]]; }

# Whether the gguf has MTP (nextn) tensors.
#   Tensor names are packed together in the gguf header, so reading the head
#   is enough. No gguf package and no python needed. For split gguf, the
#   header is in the first shard.
# Whether the gguf has MTP tensors.
#
#   Use the repository's gguf_probe.py as the primary source of truth. Tensor
#   naming varies by model (nextn / mtp / multi_token), so a home-grown
#   pattern misses some (hit in practice). Grep the header directly only when
#   gguf_probe is unusable.
#
#   ⚠️ Do not write the fallback grep as `head | grep -q`. grep exits as soon
#      as it matches, head dies of SIGPIPE, and pipefail picks up that 141 as
#      "no match". Use process substitution and look only at grep's exit code.
#   ⚠️ The dangerous direction of a misjudgment is saying "absent" (MTP is
#      silently dropped = 2x slower). If the tensors are present but
#      unsupported, the server fails to start and you will notice, so when in
#      doubt read more bytes.
MTP_DETECT_BY=""

_gguf_python() {          # find a python that can import the gguf package
  local p
  for p in "$(dirname "$LLMBENCH")/python3" "$(dirname "$LLMBENCH")/python" python3 python; do
    command -v "$p" >/dev/null 2>&1 || continue
    if "$p" -c 'import gguf' >/dev/null 2>&1; then printf '%s' "$p"; return 0; fi
  done
  return 1
}

gguf_has_mtp() {
  local f="$1" py n
  if [[ "$MTP_DETECT" != "grep" && -f "$REPO_DIR/gguf_probe.py" ]] && py="$(_gguf_python)"; then
    n="$("$py" "$REPO_DIR/gguf_probe.py" "$f" 2>/dev/null \
         | sed -n 's/.*MTP\/NextN tensors: *\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [[ -n "$n" ]]; then
      MTP_DETECT_BY="gguf_probe(${n} tensors)"
      (( n > 0 ))
      return
    fi
  fi
  MTP_DETECT_BY="grep"
  for n in "$MTP_SCAN_BYTES" "$MTP_SCAN_MAX_BYTES"; do
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > 0 )) || continue
    if LC_ALL=C grep -qaE 'blk\.[0-9]+\.(nextn|mtp|multi_token)' \
         < <(head -c "$n" "$f" 2>/dev/null); then
      return 0
    fi
  done
  return 1
}

# remove --spec-type draft-mtp (and --spec-type=draft-mtp) from SERVER_ARGS
strip_mtp_args() {
  local -a out=(); local i n=${#SERVER_ARGS[@]}
  for ((i = 0; i < n; i++)); do
    if [[ "${SERVER_ARGS[$i]}" == "--spec-type" && "${SERVER_ARGS[$((i + 1))]:-}" == "draft-mtp" ]]; then
      i=$((i + 1)); continue
    fi
    [[ "${SERVER_ARGS[$i]}" == "--spec-type=draft-mtp" ]] && continue
    out+=("${SERVER_ARGS[$i]}")
  done
  SERVER_ARGS=("${out[@]}")
}

MTP_USED="-"
build_server_args() {
  local q="$1" gguf="$2" ov
  SERVER_ARGS=(-m "$gguf" --host "$HOST" --port "$PORT"
               -ngl "$NGL" --ctx-size "$CTX" --parallel "$PARALLEL"
               --batch-size "$BATCH" --ubatch-size "$UBATCH")
  [[ -n "$FLASH_ATTN"   ]] && SERVER_ARGS+=(-fa "$FLASH_ATTN")
  [[ -n "$KV_TYPE"      ]] && SERVER_ARGS+=(-ctk "$KV_TYPE" -ctv "$KV_TYPE")
  [[ -n "$DEVICE"       ]] && SERVER_ARGS+=(--device "$DEVICE")
  [[ -n "$TENSOR_SPLIT" ]] && SERVER_ARGS+=(--tensor-split "$TENSOR_SPLIT")
  # shellcheck disable=SC2206
  [[ -n "$SERVER_EXTRA_ARGS" ]] && SERVER_ARGS+=($SERVER_EXTRA_ARGS)
  ov="$(deref "OVERRIDE_$(varname "$q")")"
  # shellcheck disable=SC2206
  [[ -n "$ov" ]] && SERVER_ARGS+=($ov)

  # --- MTP (speculative decoding) auto-detection ----------------------------
  MTP_USED="-"
  local a requested=0
  for a in "${SERVER_ARGS[@]}"; do
    [[ "$a" == "draft-mtp" || "$a" == "--spec-type=draft-mtp" ]] && requested=1
  done
  if (( requested )); then
    if [[ "$MTP_AUTO" == "1" ]] && ! gguf_has_mtp "$gguf"; then
      warn "  ⚠️ This gguf has no MTP tensors, so --spec-type draft-mtp is removed [detected by: ${MTP_DETECT_BY:-?}]"
      warn "     (with it kept, llama-server fails to start)"
      warn "     Speed is not comparable with MTP-enabled quants. Check the mtp column of the manifest"
      strip_mtp_args
      MTP_USED="no"
    else
      MTP_USED="yes"
      log "  MTP detection: ${MTP_DETECT_BY:-?}"
    fi
  fi
  return 0
}

start_server() {
  local q="$1" gguf="$2"
  SERVER_LOG="$LOG_DIR/${q}_server.log"
  build_server_args "$q" "$gguf"

  if [[ "$DRY_RUN" == "1" ]]; then
    [[ "$MTP_USED" != "-" ]] && log "  MTP: $MTP_USED"
    log "DRY-RUN: $LLAMA_SERVER ${SERVER_ARGS[*]}  > $SERVER_LOG"
    return 0
  fi

  if port_busy; then
    if [[ "$FORCE_PORT" == "1" ]]; then
      warn "Existing process on port $PORT. Killing it (--force-port)"
      pkill -f -- "--port $PORT" || true
      sleep 3
      port_busy && die "Could not free port $PORT"
    else
      die "Port $PORT is already in use. Add --force-port or stop the process manually"
    fi
  fi

  info "Starting llama-server: $q"
  [[ "$MTP_USED" != "-" ]] && log "  MTP: $MTP_USED"
  log  "  $LLAMA_SERVER ${SERVER_ARGS[*]}"
  "$LLAMA_SERVER" "${SERVER_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  local waited=0 code
  while (( waited < HEALTH_TIMEOUT )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      err "llama-server died right after startup. Tail of the log:"
      tail -n 20 "$SERVER_LOG" >&2 || true
      SERVER_PID=""
      return 1
    fi
    code="$(health_code)"
    if [[ "$code" == "200" ]]; then
      log "  /health OK (${waited}s)"
      return 0
    fi
    sleep 3; waited=$((waited + 3))
    (( waited % 60 == 0 )) && log "  Waiting for load ${waited}s (HTTP $code)"
  done
  err "/health did not pass within ${HEALTH_TIMEOUT}s"
  return 1
}

stop_server() {
  [[ "$DRY_RUN" == "1" ]] && { SERVER_PID=""; return 0; }
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping llama-server (pid $SERVER_PID)"
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    local waited=0
    while (( waited < STOP_WAIT )) && kill -0 "$SERVER_PID" 2>/dev/null; do
      sleep 1; waited=$((waited + 1))
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      warn "Did not die on SIGTERM, sending SIGKILL"
      kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
  (( VRAM_SETTLE > 0 )) && sleep "$VRAM_SETTLE"
  return 0
}

record_server_env() {
  local q="$1"
  [[ "$DRY_RUN" == "1" ]] && return 0
  "${CURL[@]}" -m 10 "http://$HOST:$PORT/props"     -o "$LOG_DIR/${q}_props.json"  || true
  "${CURL[@]}" -m 10 "http://$HOST:$PORT/v1/models" -o "$LOG_DIR/${q}_models.json" || true
  local loaded n_ctx
  loaded="$(python3 - "$LOG_DIR/${q}_models.json" <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print((d.get("data") or [{}])[0].get("id",""))
except Exception:
    print("")
PY
)"
  n_ctx="$(python3 - "$LOG_DIR/${q}_props.json" <<'PY' 2>/dev/null || true
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    g=d.get("default_generation_settings") or {}
    print(g.get("n_ctx") or d.get("n_ctx") or "")
except Exception:
    print("")
PY
)"
  log "  Loaded model: ${loaded:-?}  / n_ctx: ${n_ctx:-?}"

  # --- Check that the model really landed on the GPU -------------------------
  #   Even with -ngl 99, llama.cpp puts some layers on the CPU when VRAM is
  #   short. Generation becomes tens of times slower, and the log says
  #   nothing (hit in practice). Warn when less than half of the model file
  #   size is used in VRAM. There may be more than one GPU, so sum all GPUs
  #   (supports tensor-split).
  local gpu_used=0 gpu_total=0 gpu_lines=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_lines="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total \
                 --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$gpu_lines" ]]; then
      local idx name used total
      while IFS=, read -r idx name used total; do
        used="$(printf '%s' "$used" | tr -d ' ')"
        total="$(printf '%s' "$total" | tr -d ' ')"
        [[ "$used"  =~ ^[0-9]+$ ]] || continue
        [[ "$total" =~ ^[0-9]+$ ]] || total=0
        log "  VRAM GPU${idx// /}:$(printf '%s' "$name") ${used}/${total} MiB"
        gpu_used=$((gpu_used + used)); gpu_total=$((gpu_total + total))
      done <<< "$gpu_lines"
      # du measures disk usage, so don't use it (a sparse file reports 0). Look at the real size.
      local file_bytes file_mib
      file_bytes="$(stat -c %s "$CURRENT_GGUF" 2>/dev/null || stat -f %z "$CURRENT_GGUF" 2>/dev/null || echo 0)"
      file_mib=$(( file_bytes / 1048576 ))
      if (( gpu_used > 0 )); then
        log "  VRAM total: ${gpu_used}/${gpu_total} MiB"
      fi
      if [[ -n "$file_mib" ]] && (( gpu_used * 2 < file_mib )); then
        warn "  ⚠️ VRAM in use (total ${gpu_used} MiB) is less than half of the model size (${file_mib} MiB)."
        warn "     It probably does not fully fit on the GPU (= CPU execution, extremely slow)."
        warn "     Consider setting DEVICE / TENSOR_SPLIT, lowering --ctx-size, or KV_TYPE=q8_0."
        warn "     Log: $SERVER_LOG"
      fi
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$q" "${loaded:-?}" "${n_ctx:-?}" \
         "${gpu_used:-?}" "${gpu_total:-?}" "${MTP_USED:--}" >> "$OUT_ROOT/manifest_${RUN_ID}.tsv"
}

# =============================================================================
#  config (seed disabling)
# =============================================================================
# effective ctx per quant (the common CTX, with SERVER_EXTRA_ARGS / OVERRIDE_* overriding — later wins)
effective_ctx() {
  local q="$1" ctx="$CTX" tok prev="" s
  s="$SERVER_EXTRA_ARGS $(deref "OVERRIDE_$(varname "$q")")"
  for tok in $s; do
    case "$prev" in
      --ctx-size|-c) [[ "$tok" =~ ^[0-9]+$ ]] && ctx="$tok" ;;
    esac
    prev="$tok"
  done
  printf '%s' "$ctx"
}

# read the max_tokens written in the config (inside the MODEL_KEY block)
config_max_tokens() {
  awk -v key="$MODEL_KEY" '
    BEGIN { kpat = "^  (\"" key "\"|" key "):[[:space:]]*$" }   # quoted or plain model key
    /^[^[:space:]#]/                { inblk = 0 }
    $0 ~ kpat { inblk = 1; next }
    inblk && /^  [A-Za-z0-9_.\-]+:[[:space:]]*$/ { inblk = 0 }
    inblk && match($0, /^[[:space:]]*max_tokens:[[:space:]]*[0-9]+/) {
      s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); print s; exit
  }
  ' "$CONFIG"
}

# safe max_tokens for the effective ctx. Same rule as gguf_plan.py:
#   min(3/4 of ctx rounded down to units of 1024, MAX_TOKENS_CAP)
auto_max_tokens() {
  local ctx="$1" tq=$(( $1 * 3 / 4 / 1024 * 1024 ))
  if (( tq <= MAX_TOKENS_CAP )); then printf '%s' "$tq"; else printf '%s' "$MAX_TOKENS_CAP"; fi
}

# Build a temp config per quant × (drop the seed or not). The original config.yaml is read-only.
CONFIG_NOTE=""
CONFIG_PATH=""
config_for() {   # results go into CONFIG_PATH / CONFIG_NOTE (do not subshell)
  local q="$1" runs="$2" suite="${3:-}"
  local strip=0 ctx mt cur out tag rmt="" mt_eff
  CONFIG_NOTE=""
  [[ -n "$suite" ]] && rmt="$(deref "REASONING_MAX_TOKENS_$(varname "$suite")")"
  if [[ -n "$rmt" && ! "$rmt" =~ ^[0-9]+$ ]]; then
    die "REASONING_MAX_TOKENS_$(varname "$suite")=$rmt is not an integer"
  fi
  [[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

  if [[ "$STRIP_SEED_FOR_MULTIRUN" == "1" && "${runs:-1}" -gt 1 ]]; then strip=1; fi

  mt=""
  if [[ "$ADJUST_MAX_TOKENS" == "1" ]]; then
    ctx="$(effective_ctx "$q")"
    if [[ -n "$MAX_TOKENS" ]]; then mt="$MAX_TOKENS"; else mt="$(auto_max_tokens "$ctx")"; fi
    cur="$(config_max_tokens)"
    # only rewrite when lowering (raising ctx never raises it on its own)
    if [[ -n "$cur" ]] && (( cur <= mt )); then mt=""; fi
  fi

  if [[ -n "$rmt" ]]; then
    mt_eff="${mt:-$(config_max_tokens)}"
    if [[ -n "$mt_eff" ]] && (( rmt >= mt_eff )); then
      warn "  REASONING_MAX_TOKENS_$(varname "$suite")=$rmt is >= max_tokens=$mt_eff. No room left to write the answer"
    fi
  fi

  if (( strip == 0 )) && [[ -z "$mt" && -z "$rmt" ]]; then
    CONFIG_PATH="$CONFIG"; return 0
  fi

  tag="$(varname "$q")"; (( strip )) && tag="${tag}_noseed"
  [[ -n "$rmt" ]] && tag="${tag}_$(varname "$suite")_r${rmt}"
  out="$OUT_ROOT/config_${tag}.yaml"
  # whether the MODEL_KEY block already has reasoning_max_tokens (if not, add it right after the key line)
  local rhas=0
  if [[ -n "$rmt" ]] && awk -v key="$MODEL_KEY" '
      BEGIN { kpat = "^  (\"" key "\"|" key "):[[:space:]]*$" }
      /^[^[:space:]#]/                { inblk = 0 }
      $0 ~ kpat { inblk = 1; next }
      inblk && /^  [A-Za-z0-9_.\-]+:[[:space:]]*$/ { inblk = 0 }
      inblk && /^[[:space:]]*reasoning_max_tokens:/ { found = 1 }
      END { exit !found }' "$CONFIG"; then
    rhas=1
  fi
  awk -v key="$MODEL_KEY" -v strip="$strip" -v mt="$mt" -v rmt="$rmt" -v rhas="$rhas" '
    BEGIN { kpat = "^  (\"" key "\"|" key "):[[:space:]]*$" }
    /^[^[:space:]#]/                { inblk = 0 }
    $0 ~ kpat {
      inblk = 1; print
      if (rmt != "" && rhas == 0) print "    reasoning_max_tokens: " rmt "   # sweep.sh: added for this suite"
      next
    }
    inblk && /^  [A-Za-z0-9_.\-]+:[[:space:]]*$/ { inblk = 0 }
    inblk && rmt != "" && match($0, /^[[:space:]]*reasoning_max_tokens:[[:space:]]*[0-9]+/) {
      indent = $0; sub(/[^[:space:]].*$/, "", indent)
      print indent "reasoning_max_tokens: " rmt "   # sweep.sh: rewritten for this suite"
      next
    }
    inblk && strip == 1 && /^[[:space:]]*seed:/ {
      print "#" $0 "   # sweep.sh: disabled because runs>1"; next
    }
    inblk && mt != "" && match($0, /^[[:space:]]*max_tokens:[[:space:]]*[0-9]+/) {
      indent = $0; sub(/[^[:space:]].*$/, "", indent)
      print indent "max_tokens: " mt "   # sweep.sh: lowered to fit the effective ctx"
      next
    }
    { print }
  ' "$CONFIG" > "$out"

  if (( strip )); then CONFIG_NOTE="seed disabled"; fi
  if [[ -n "$mt" ]]; then
    CONFIG_NOTE="${CONFIG_NOTE:+$CONFIG_NOTE / }max_tokens ${cur:-?} → $mt (ctx $ctx)"
  fi
  if [[ -n "$rmt" ]]; then
    CONFIG_NOTE="${CONFIG_NOTE:+$CONFIG_NOTE / }reasoning_max_tokens → $rmt ($suite)"
    grep -Eq "^[[:space:]]*reasoning_max_tokens:[[:space:]]*${rmt}([^0-9]|$)" "$out" \
      || die "could not write reasoning_max_tokens: $rmt to the temp config ($out)"
  fi
  CONFIG_PATH="$out"
}

# =============================================================================
#  state (resume)
# =============================================================================
STATE_FILE=""
state_key()  { printf '%s\t%s' "$1" "$2"; }
state_done() {
  [[ "$RESUME" == "1" ]] || return 1
  [[ -f "$STATE_FILE" ]] || return 1
  awk -F'\t' -v q="$1" -v s="$2" '$1==q && $2==s && $3=="ok" {found=1} END{exit !found}' "$STATE_FILE"
}
state_put() {  # quant suite status seconds results_path
  [[ "$DRY_RUN" == "1" ]] && return 0    # DRY-RUN does not dirty the state
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:-}" >> "$STATE_FILE"
}
summary_put() {  # quant suite status seconds results_path
  [[ "$DRY_RUN" == "1" ]] && return 0
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "${5:-}" >> "$SUMMARY"
}

# =============================================================================
#  Post-run processing
# =============================================================================
# Classify sec / secaug results by failure mode and save them. A failure here does not stop the sweep.
post_classify() {
  local q="$1" s="$2" out py
  [[ "$POST_SEC_CLASSIFY" == "1" ]] || return 0
  case "$s" in sec|secaug) ;; *) return 0 ;; esac
  [[ -n "$RESULT_PATH" ]] || { warn "  Classification skipped: result path unknown"; return 0; }
  local rp="$RESULT_PATH"
  [[ "$rp" = /* ]] || rp="$REPO_DIR/$rp"
  [[ -f "$rp" ]] || { warn "  Classification skipped: $rp not found"; return 0; }
  py="$(dirname "$LLMBENCH")/python"; [[ -x "$py" ]] || py=python3
  out="$LOG_DIR/${q}_${s}_classify.txt"
  if "$py" "$REPO_DIR/tools/sec_classify.py" "$rp" > "$out" 2>&1; then
    log "  Classified: $out"
    grep -E '^Success rate|^Failure' "$out" | sed 's/^/     /' || true
  else
    warn "  Classification failed (sweep continues): $out"
  fi
}

# =============================================================================
#  Run one suite
# =============================================================================
run_suite() {  # quant suite -> 0/1, result json in RESULT_PATH
  local q="$1" s="$2"
  local runs args label logf cfg rc started elapsed
  runs="$(suite_runs "$s")"; args="$(suite_args "$s")"
  label="${LABEL_PREFIX:+${LABEL_PREFIX}-}${q}-${s}"
  logf="$LOG_DIR/${q}_${s}.log"
  RESULT_PATH=""

  config_for "$q" "$runs" "$s"
  cfg="$CONFIG_PATH"
  if [[ "$cfg" != "$CONFIG" ]]; then
    log "  config: $(basename "$cfg")  [${CONFIG_NOTE:-temp config}]"
  fi

  # llmbench progress (print) goes to stdout. Piped to a tee, Python switches
  # to block buffering and shows not a single line until several KB pile up
  # (measured: the screen looks frozen the whole time the first L6 task is
  # being generated). PYTHONUNBUFFERED=1 streams line by line. Use stdbuf too
  # when it exists.
  local -a unbuf=(env PYTHONUNBUFFERED=1)
  if command -v stdbuf >/dev/null 2>&1; then unbuf+=(stdbuf -oL -eL); fi

  local cmd=("${unbuf[@]}" "$LLMBENCH" run --model "$MODEL_KEY" --config "$cfg"
             --tasks-dir "$TASKS_DIR" --output "$RESULTS_DIR"
             --label "$label" --runs "$runs")
  # shellcheck disable=SC2206
  cmd+=($args)
  [[ -n "$CONCURRENCY" ]] && cmd+=(--concurrency "$CONCURRENCY")
  [[ "$SKIP_PREFLIGHT" == "1" ]] && cmd+=(--skip-preflight)
  if (( SUITE_TIMEOUT > 0 )) && command -v timeout >/dev/null 2>&1; then
    cmd=(timeout "${SUITE_TIMEOUT}s" "${cmd[@]}")
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY-RUN: (cd $REPO_DIR && ${cmd[*]})"
    return 0
  fi

  info "  ▶ $q / $s  (runs=$runs $args)"
  started=$SECONDS
  set +e
  ( cd "$REPO_DIR" && "${cmd[@]}" ) 2>&1 | tee "$logf"
  rc=${PIPESTATUS[0]}
  set -e
  elapsed=$((SECONDS - started))

  RESULT_PATH="$(grep -a '^Results: ' "$logf" | tail -1 | sed 's/^Results: //' || true)"
  if [[ -z "$RESULT_PATH" ]]; then
    RESULT_PATH="$(find "$RESULTS_DIR" -maxdepth 1 -name "*_${label}_results.json" 2>/dev/null | sort | tail -1)"
  fi

  if [[ $rc -eq 0 ]]; then
    log "  ✅ $q / $s  $(hms "$elapsed")  ${RESULT_PATH:-(result path unknown)}"
    post_classify "$q" "$s"
  else
    err "  ❌ $q / $s  rc=$rc  $(hms "$elapsed")  log: $logf"
    if [[ $rc -eq 2 ]] && grep -qa "preflight is FAIL" "$logf" 2>/dev/null; then
      err "     preflight is FAIL. See section \"B. Cross-check effective values\" in the"
      err "     log and fix the config or the launch args (to run anyway with that"
      err "     verdict accepted: SKIP_PREFLIGHT=1)"
    fi
  fi
  SUITE_ELAPSED=$elapsed
  return $rc
}

# =============================================================================
#  main
# =============================================================================
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$OUT_ROOT/logs/$RUN_ID"
STATE_FILE="$OUT_ROOT/sweep_state${LABEL_PREFIX:+_${LABEL_PREFIX}}.tsv"
SUMMARY="$OUT_ROOT/summary_${RUN_ID}.tsv"

mapfile -t QUANT_LIST < <(resolve_quants)
mapfile -t SUITE_LIST < <(resolve_suites)
QUANT_LIST=("${QUANT_LIST[@]:-}"); SUITE_LIST=("${SUITE_LIST[@]:-}")
[[ -n "${QUANT_LIST[0]:-}" ]] || die "0 target quants (QUANTS=$QUANTS)"
[[ -n "${SUITE_LIST[0]:-}" ]] || die "0 target suites (all RUN_*=0, or --suites is empty)"

if [[ "$DO_LIST" == "1" ]]; then
  echo "Config file: ${SWEEP_CONF_LOADED:-(none loaded — script defaults)}"
  echo "REPO_DIR  : $REPO_DIR"
  echo "LLMBENCH  : $LLMBENCH"
  echo "CONFIG    : $CONFIG"
  echo "OUT_ROOT  : $OUT_ROOT"
  echo "RESULTS   : $RESULTS_DIR"
  echo "MODEL_DIR : $MODEL_DIR"
  echo "MODEL_PFX : $MODEL_PREFIX"
  echo "LABEL_PFX : ${LABEL_PREFIX:-(none)}"
  echo "Quants    : ${QUANT_LIST[*]}"
  echo "Suites    :"
  for s in "${SUITE_LIST[@]}"; do
    r="$(deref "REASONING_MAX_TOKENS_$(varname "$s")")"
    printf '  %-8s runs=%-3s %s%s\n' "$s" "$(suite_runs "$s")" "$(suite_args "$s")" \
      "${r:+   [reasoning_max_tokens=$r]}"
  done
  echo "Total     : $(( ${#QUANT_LIST[@]} * ${#SUITE_LIST[@]} )) runs"
  exit 0
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
touch "$STATE_FILE"

# prevent a double start
LOCK_FILE="$OUT_ROOT/.sweep.lock"
if [[ "$DRY_RUN" != "1" ]]; then
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    flock -n 9 || die "Another sweep is running ($LOCK_FILE)"
    echo "$$" >&9
  fi
fi

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -n "$SERVER_PID" ]] && { warn "Cleanup: stopping llama-server"; stop_server; }
  exit $rc
}
trap cleanup EXIT INT TERM

info "════════ llmbench sweep $RUN_ID ════════"
log "Config    : ${SWEEP_CONF_LOADED:-(none loaded — script defaults)}"
log "Quants    : ${QUANT_LIST[*]}"
log "Suites    : ${SUITE_LIST[*]}"
log "Logs      : $LOG_DIR"
[[ "$DRY_RUN" == "1" ]] && warn "DRY-RUN mode (nothing is executed)"

mkdir -p "$OUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then
  printf 'quant\tmodel_id\tn_ctx\tvram_used_mib\tvram_total_mib\tmtp\n' > "$OUT_ROOT/manifest_${RUN_ID}.tsv"
  printf 'quant\tsuite\tstatus\tseconds\tresults\n' > "$SUMMARY"
fi

TOTAL=0; OK=0; NG=0; SKIP=0
SWEEP_START=$SECONDS

for q in "${QUANT_LIST[@]}"; do
  echo
  info "──────── Quant: $q ────────"

  # does anything remain to run for this model (resume)
  pending=()
  for s in "${SUITE_LIST[@]}"; do
    if state_done "$q" "$s"; then
      log "  ⏭ $q / $s already completed → skip"
      SKIP=$((SKIP + 1)); TOTAL=$((TOTAL + 1))
      summary_put "$q" "$s" "skipped" 0 ""
    else
      pending+=("$s")
    fi
  done
  if [[ ${#pending[@]} -eq 0 ]]; then
    log "  All suites for this quant are completed. Not starting the server"
    continue
  fi

  gguf="$(gguf_path "$q" || true)"
  if [[ -z "$gguf" ]]; then
    err "  gguf not found: ${MODEL_PREFIX}-${q}.gguf ($MODEL_DIR)"
    for s in "${pending[@]}"; do
      TOTAL=$((TOTAL + 1)); NG=$((NG + 1))
      state_put "$q" "$s" "no_model" 0 ""
      summary_put "$q" "$s" "no_model" 0 ""
    done
    [[ "$KEEP_GOING" == "1" ]] && continue || exit 1
  fi
  CURRENT_GGUF="$gguf"
  log "  gguf: $gguf ($(du -h "$gguf" 2>/dev/null | cut -f1))"

  if ! start_server "$q" "$gguf"; then
    err "  Server failed to start → skipping this quant"
    stop_server
    for s in "${pending[@]}"; do
      TOTAL=$((TOTAL + 1)); NG=$((NG + 1))
      state_put "$q" "$s" "server_fail" 0 ""
      summary_put "$q" "$s" "server_fail" 0 ""
    done
    [[ "$KEEP_GOING" == "1" ]] && continue || exit 1
  fi
  record_server_env "$q"

  for s in "${pending[@]}"; do
    TOTAL=$((TOTAL + 1))
    SUITE_ELAPSED=0
    if run_suite "$q" "$s"; then
      OK=$((OK + 1))
      state_put "$q" "$s" "ok" "$SUITE_ELAPSED" "$RESULT_PATH"
      summary_put "$q" "$s" "ok" "$SUITE_ELAPSED" "$RESULT_PATH"
    else
      NG=$((NG + 1))
      state_put "$q" "$s" "failed" "$SUITE_ELAPSED" ""
      summary_put "$q" "$s" "failed" "$SUITE_ELAPSED" ""
      if [[ "$KEEP_GOING" != "1" ]]; then
        stop_server
        die "Aborted on suite failure (--stop-on-error)"
      fi
    fi
  done

  stop_server
done

# --- Summary ------------------------------------------------------------------
echo
info "════════ Summary ($(hms $((SECONDS - SWEEP_START)))) ════════"
if [[ "$DRY_RUN" != "1" ]]; then
  {
    printf '%-14s %-9s %-11s %-9s %s\n' "QUANT" "SUITE" "STATUS" "TIME" "RESULTS"
    tail -n +2 "$SUMMARY" | while IFS=$'\t' read -r q s st sec res; do
      printf '%-14s %-9s %-11s %-9s %s\n' "$q" "$s" "$st" "$(hms "${sec:-0}")" "${res:-}"
    done
  } | sed 's/^/  /'
  echo
  log "Total $TOTAL  /  ok $OK  /  failed $NG  /  skipped $SKIP"
  log "Summary : $SUMMARY"
  log "manifest: $OUT_ROOT/manifest_${RUN_ID}.tsv"
  log "state   : $STATE_FILE  (--resume skips completed work)"
  echo
  log "Cross-suite comparison reports can be built with:"
  for s in "${SUITE_LIST[@]}"; do
    files="$(awk -F'\t' -v s="$s" '$3=="ok" && $2==s {print $5}' "$SUMMARY" | tr '\n' ' ')"
    [[ -n "${files// /}" ]] && echo "  $LLMBENCH compare $files --name ${s}_by_quant"
  done
fi

[[ $NG -gt 0 ]] && exit 1
exit 0
