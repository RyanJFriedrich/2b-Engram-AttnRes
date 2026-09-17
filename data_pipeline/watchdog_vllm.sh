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
STARTUP_TIMEOUT_SEC=300   # Allow up to 5 minutes for GGUF dequantization & KV initialization
COOL_DOWN_SEC=15

trap "echo ''; log 'SIGINT/SIGTERM received. Exiting Watchdog supervisor.'; exit 0" INT TERM

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
    log "Stopping vLLM engine processes (APIServer + EngineCore)..."
    # 1. Release listening TCP socket
    fuser -k -9 8000/tcp 2>/dev/null || true

    # 2. Terminate internal worker engine cores
    pkill -9 -f "EngineCore" 2>/dev/null || true
    pkill -9 -f "VLLM::" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true

    # 3. Kill python vllm serve process, explicitly excluding shell scripts and self
    for pid in $(pgrep -f "python.*vllm|vllm serve" 2>/dev/null || true); do
        if [ "$pid" != "$$" ] && [ "$pid" != "$PPID" ]; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 3
}

wait_for_healthy() {
    local elapsed=0
    log "Waiting for vLLM initialization and /health 200 OK (up to ${STARTUP_TIMEOUT_SEC}s)..."
    while [ $elapsed -lt $STARTUP_TIMEOUT_SEC ]; do
        sleep 5
        elapsed=$((elapsed + 5))

        if is_healthy; then
            log "vLLM is ONLINE and HEALTHY (took ~${elapsed}s). Resuming monitoring."
            return 0
        fi

        if [ $((elapsed % 30)) -eq 0 ]; then
            log "Still initializing model / KV cache... (${elapsed}s elapsed)"
        fi
    done

    log "WARNING: vLLM did not become healthy within ${STARTUP_TIMEOUT_SEC}s."
    return 1
}

restart_vllm() {
    kill_vllm
    log "Cooling down ${COOL_DOWN_SEC}s for Windows WDDM to reclaim VRAM..."
    sleep "$COOL_DOWN_SEC"

    # Check if an interactive start_vllm.sh loop is already running in another terminal
    local runner_pids
    runner_pids=$(pgrep -f "start_vllm\.sh" 2>/dev/null || true)
    local external_runner=0
    for rpid in $runner_pids; do
        if [ "$rpid" != "$$" ] && [ "$rpid" != "$PPID" ]; then
            external_runner=1
            break
        fi
    done

    if [ "$external_runner" -eq 1 ]; then
        log "Active start_vllm.sh runner detected. The runner loop will automatically restart vLLM."
    else
        log "No active start_vllm.sh runner found. Starting background vLLM instance via $START_SCRIPT..."
        nohup bash "$START_SCRIPT" > "$LOG_FILE" 2>&1 &
        local spawned_pid=$!
        log "Background vLLM spawned (PID: $spawned_pid)."
    fi

    wait_for_healthy
}

log "======================================================================"
log "Starting vLLM Watchdog Supervisor"
log "Target Health URL: $HEALTH_URL"
log "Watch Pattern:     dmesg ['$FAULT_PATTERN']"
log "Startup Budget:    ${STARTUP_TIMEOUT_SEC}s"
log "======================================================================"

last_fault_count=$(get_fault_count)
log "Initial WDDM fault count: $last_fault_count"

# Check if vLLM is ALREADY running and healthy
if is_healthy; then
    log "Detected existing healthy vLLM server on port 8000. Attaching monitor (WILL NOT KILL)."
else
    log "No active healthy server detected on port 8000. Ensuring server is running..."
    restart_vllm
fi

consecutive_health_fails=0

while true; do
    sleep "$CHECK_INTERVAL_SEC"

    # 1. Check for hard WDDM driver fault in dmesg
    current_fault_count=$(get_fault_count)
    if [ "$current_fault_count" -gt "$last_fault_count" ]; then
        log "Kernel WDDM fault event logged (count: $last_fault_count -> $current_fault_count)."
        # Verify if vLLM engine actually died or became unresponsive
        if ! is_healthy || ! pgrep -f "EngineCore" > /dev/null 2>&1; then
            log "CRITICAL: vLLM is confirmed DEAD or UNRESPONSIVE following WDDM fault. Rebooting..."
            restart_vllm
            last_fault_count=$(get_fault_count)
            consecutive_health_fails=0
            continue
        else
            log "vLLM is still ONLINE and HEALTHY despite kernel warning. Skipping reboot."
            last_fault_count=$current_fault_count
        fi
    fi

    # 2. Check HTTP health endpoint
    if is_healthy; then
        consecutive_health_fails=0
    else
        consecutive_health_fails=$((consecutive_health_fails + 1))
        log "Health check probe failed ($consecutive_health_fails / 3)"
        if [ "$consecutive_health_fails" -ge 3 ]; then
            log "vLLM unresponsive for $((consecutive_health_fails * CHECK_INTERVAL_SEC))s. Restarting..."
            restart_vllm
            last_fault_count=$(get_fault_count)
            consecutive_health_fails=0
        fi
    fi
done
