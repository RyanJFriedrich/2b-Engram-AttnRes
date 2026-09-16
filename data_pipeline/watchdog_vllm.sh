#!/usr/bin/env bash
# ==============================================================================
# vLLM Autonomous Supervisor & WDDM Fault Watchdog (WSL2)
# Detects hard Windows WDDM driver faults and auto-reboots vLLM
# ==============================================================================

set -u

START_SCRIPT="/home/ophidian/start_vllm.sh"
HEALTH_URL="http://127.0.0.1:8000/health"
CHECK_INTERVAL_SEC=10
FAULT_PATTERN="dxgkio_make_resident: Ioctl failed: -12"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Watchdog] $*"
}

get_fault_count() {
    dmesg | grep -c "$FAULT_PATTERN" 2>/dev/null || echo 0
}

kill_vllm() {
    log "Terminating all vLLM processes..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 3
}

cleanup() {
    log "Watchdog received interrupt/terminate signal. Shutting down vLLM..."
    kill_vllm
    exit 0
}

trap cleanup SIGINT SIGTERM

start_server() {
    kill_vllm
    log "Starting vLLM server via $START_SCRIPT..."
    bash "$START_SCRIPT" &
    VLLM_PID=$!
    log "vLLM spawned (PID: $VLLM_PID). Waiting for /health to become ready..."

    local ready=0
    for i in $(seq 1 60); do
        if curl -s -f --max-time 3 "$HEALTH_URL" > /dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done

    if [ $ready -eq 1 ]; then
        log "vLLM is ONLINE and HEALTHY (took ~$((i * 2))s)."
    else
        log "WARNING: vLLM did not report ready within 120s! Monitoring continues..."
    fi
}

log "======================================================================"
log "Starting vLLM Supervisor with Hardware Fault Detection"
log "Watch target: Kernel log ['$FAULT_PATTERN'] + HTTP ['$HEALTH_URL']"
log "======================================================================"

last_fault_count=$(get_fault_count)
log "Initial WDDM fault baseline: $last_fault_count"

# Initial launch
start_server

consecutive_health_fails=0

while true; do
    sleep "$CHECK_INTERVAL_SEC"

    # 1. Check for hard WDDM kernel driver faults
    current_fault_count=$(get_fault_count)
    if [ "$current_fault_count" -gt "$last_fault_count" ]; then
        log "CRITICAL: Hard WDDM driver fault detected in dmesg!"
        log "Fault count increased: $last_fault_count -> $current_fault_count"
        log "Cooling down 10s for Windows memory manager to reclaim pages..."
        kill_vllm
        sleep 10
        last_fault_count=$(get_fault_count)
        start_server
        consecutive_health_fails=0
        continue
    fi

    # 2. Check HTTP health endpoint
    if curl -s -f --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
        consecutive_health_fails=0
    else
        consecutive_health_fails=$((consecutive_health_fails + 1))
        log "Health check failed ($consecutive_health_fails / 3)"
        if [ "$consecutive_health_fails" -ge 3 ]; then
            log "Server unresponsive for $((consecutive_health_fails * CHECK_INTERVAL_SEC))s. Rebooting vLLM..."
            kill_vllm
            sleep 10
            last_fault_count=$(get_fault_count)
            start_server
            consecutive_health_fails=0
        fi
    fi
done
