#!/usr/bin/env bash
# ==============================================================================
# vLLM Autonomous Supervisor & WDDM Fault Watchdog (WSL2)
# ==============================================================================

set -u

START_SCRIPT="/home/ophidian/start_vllm.sh"
LOG_FILE="/home/ophidian/vllm.log"
HEALTH_URL="http://127.0.0.1:8000/health"
CHECK_INTERVAL_SEC=10
FAULT_PATTERN="dxgkio_make_resident: Ioctl failed: -12"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Watchdog] $*"
}

get_fault_count() {
    dmesg | grep -c "$FAULT_PATTERN" 2>/dev/null || echo 0
}

is_healthy() {
    curl -s -f --max-time 4 "$HEALTH_URL" > /dev/null 2>&1
}

kill_vllm() {
    log "Stopping vLLM processes (APIServer + EngineCore)..."
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "vllm" 2>/dev/null || true
    sleep 5
}

start_server() {
    kill_vllm
    log "Starting vLLM server via $START_SCRIPT (logging to $LOG_FILE)..."
    nohup bash "$START_SCRIPT" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!
    log "vLLM spawned (PID: $VLLM_PID). Waiting for model initialization (up to 90s)..."

    local ready=0
    for i in $(seq 1 30); do
        sleep 3
        if is_healthy; then
            ready=1
            break
        fi
    done

    if [ $ready -eq 1 ]; then
        log "vLLM is ONLINE and HEALTHY (took ~$((i * 3))s)."
    else
        log "WARNING: vLLM not ready after 90s. Check logs at: $LOG_FILE"
    fi
}

log "======================================================================"
log "Starting vLLM Watchdog Supervisor"
log "Target Health URL: $HEALTH_URL"
log "Watch Pattern:     dmesg ['$FAULT_PATTERN']"
log "======================================================================"

last_fault_count=$(get_fault_count)
log "Initial WDDM fault count: $last_fault_count"

# Check if vLLM is ALREADY running and healthy
if is_healthy; then
    log "Detected existing healthy vLLM server on port 8000. Attaching monitor (WILL NOT KILL)."
else
    log "No active server detected on port 8000. Starting fresh vLLM instance..."
    start_server
fi

consecutive_health_fails=0

while true; do
    sleep "$CHECK_INTERVAL_SEC"

    # 1. Check for hard WDDM driver fault in dmesg
    current_fault_count=$(get_fault_count)
    if [ "$current_fault_count" -gt "$last_fault_count" ]; then
        log "CRITICAL: Hard WDDM driver fault detected in kernel log!"
        log "Fault count increased: $last_fault_count -> $current_fault_count"
        log "Cooling down 10s for Windows WDDM to reclaim VRAM..."
        kill_vllm
        sleep 10
        last_fault_count=$(get_fault_count)
        start_server
        consecutive_health_fails=0
        continue
    fi

    # 2. Check HTTP health endpoint
    if is_healthy; then
        consecutive_health_fails=0
    else
        consecutive_health_fails=$((consecutive_health_fails + 1))
        log "Health check probe failed ($consecutive_health_fails / 3)"
        if [ "$consecutive_health_fails" -ge 3 ]; then
            log "vLLM unresponsive for $((consecutive_health_fails * CHECK_INTERVAL_SEC))s. Rebooting..."
            kill_vllm
            sleep 10
            last_fault_count=$(get_fault_count)
            start_server
            consecutive_health_fails=0
        fi
    fi
done
