#!/bin/sh
set -eu

LOG_DIR="${RESOURCE_LOG_DIR:-/app/storage}"
MEM_LOG="${LOG_DIR}/memory.log"
PS_LOG="${LOG_DIR}/ps.log"
INTERVAL="${RESOURCE_LOG_INTERVAL:-5}"

mkdir -p "$LOG_DIR"

while true; do
  now="$(date -Is)"
  mem_current="$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)"
  mem_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0)"
  printf "%s current=%s max=%s\n" "$now" "$mem_current" "$mem_max" >> "$MEM_LOG"

  if command -v ps >/dev/null 2>&1; then
    {
      printf "\n%s\n" "$now"
      ps -eo pid,comm,rss,pmem --sort=-rss | head -n 20
    } >> "$PS_LOG" 2>/dev/null
  fi

  sleep "$INTERVAL"
done
