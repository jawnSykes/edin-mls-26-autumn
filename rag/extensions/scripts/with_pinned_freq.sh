#!/bin/bash
# Run a command with the benchmark core's clock pinned, then restore the previous settings.
#   - governor -> performance on one core (BENCH_CPU, default 0; match bench.py --cpu)
#   - turbo off (system-wide), so the clock doesn't drift with temperature
# Restores on normal exit, error and Ctrl-C (not on SIGKILL; a reboot resets these anyway).
# Needs sudo (local machine only; bench.py itself never needs root). Keep the laptop on AC.
#
#   extensions/scripts/with_pinned_freq.sh ~/mls-rag-env/bin/python extensions/scripts/bench.py ...
#   - several cores (e.g. multi-threaded builds): BENCH_CPUS="0 2" (space-separated; overrides BENCH_CPU)
set -u

CPU="${BENCH_CPU:-0}"
CPUS="${BENCH_CPUS:-$CPU}"
NO_TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo   # intel_pstate only

if [[ $# -eq 0 ]]; then echo "usage: $0 <command...>" >&2; exit 2; fi
declare -A GOV EPP
for c in $CPUS; do
    P="/sys/devices/system/cpu/cpu${c}/cpufreq"
    [[ -r "$P/scaling_governor" ]] || { echo "no cpufreq for cpu$c at $P" >&2; exit 1; }
    GOV[$c]="$(cat "$P/scaling_governor")"
    EPP[$c]="$(cat "$P/energy_performance_preference" 2>/dev/null || true)"
done
TURBO="$(cat "$NO_TURBO" 2>/dev/null || true)"

put() { echo "$1" | sudo tee "$2" >/dev/null; }

restore() {
    for c in $CPUS; do
        P="/sys/devices/system/cpu/cpu${c}/cpufreq"
        put "${GOV[$c]}" "$P/scaling_governor"
        # EPP can only be written under powersave; the governor change above makes that possible
        [[ -n "${EPP[$c]}" ]] && put "${EPP[$c]}" "$P/energy_performance_preference"
        echo "[with_pinned_freq] restored cpu$c governor=${GOV[$c]} epp=${EPP[$c]:-n/a}" >&2
    done
    [[ -n "$TURBO" ]] && put "$TURBO" "$NO_TURBO"
    echo "[with_pinned_freq] restored no_turbo=${TURBO:-n/a}" >&2
}

sudo -v || exit 1            # ask for the password once, before anything changes
trap restore EXIT
trap 'exit 130' INT TERM     # make Ctrl-C / kill go through the EXIT trap

for c in $CPUS; do put performance "/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor"; done
[[ -n "$TURBO" ]] && put 1 "$NO_TURBO"
for c in $CPUS; do
    echo "[with_pinned_freq] cpu$c governor=$(cat "/sys/devices/system/cpu/cpu${c}/cpufreq/scaling_governor")" >&2
done
echo "[with_pinned_freq] no_turbo=$(cat "$NO_TURBO" 2>/dev/null || echo n/a)" >&2

export BENCH_CPU="${CPUS%% *}" BENCH_CPUS="$CPUS"
"$@"
