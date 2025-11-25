#!/bin/bash
# TPU-SMI: Simple real-time TPU monitoring
# Usage: ./tpu-smi.sh [interval_seconds]

INTERVAL=${1:-5}  # Default 5s refresh

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Initialize
clear
tput civis  # Hide cursor

# Cleanup on exit
trap 'tput cnorm; clear; exit' INT TERM EXIT

# Main loop
while true; do
    # Clear entire screen properly
    printf '\033[2J\033[H'
    
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    echo -e "${BOLD}${CYAN}                    TPU-SMI Monitor${NC}"
    echo -e "${BOLD}${CYAN}================================================================${NC}"
    date "+%a %b %d %H:%M:%S %Y"
    echo ""
    
    # TPU Process Status
    echo -e "${BOLD}${BLUE}TPU Process Status:${NC}"
    tpu-info -p 2>&1 | grep -v "WARNING" | grep -v "unavailable"
    echo ""
    
    # TPU Metrics
    echo -e "${BOLD}${BLUE}TPU Metrics:${NC}"
    HBM=$(tpu-info --metric hbm_usage 2>&1 | grep -v "WARNING" | grep -v "unavailable" | grep -v "Runtime" | grep -v "information\." | grep -v "^╰" | grep -v "^│.*information")
    DUTY=$(tpu-info --metric duty_cycle_percent 2>&1 | grep -v "WARNING" | grep -v "unavailable" | grep -v "Runtime" | grep -v "information\." | grep -v "^╰" | grep -v "^│.*information")
    
    if [ -n "$HBM" ]; then
        echo "$HBM"
        echo ""
    fi
    
    if [ -n "$DUTY" ]; then
        echo "$DUTY"
        echo ""
    fi
    
    if [ -z "$HBM" ] && [ -z "$DUTY" ]; then
        echo -e "  ${YELLOW}TPU idle - No active workload${NC}"
        TPU_TYPE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type" -H "Metadata-Flavor: Google" 2>/dev/null)
        echo -e "  ${GREEN}Type: $TPU_TYPE${NC}"
        echo ""
    fi
    
    # Host System
    echo -e "${BOLD}${BLUE}Host System:${NC}"
    CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{printf "%.1f", 100 - $1}')
    MEM=$(free -h | awk '/^Mem:/ {printf "%s / %s", $3, $2}')
    echo -e "  ${CYAN}CPU:${NC} ${YELLOW}${CPU}%${NC}"
    echo -e "  ${CYAN}Memory:${NC} ${YELLOW}$MEM${NC}"
    echo ""
    echo -e "${BOLD}Press Ctrl+C to exit | Refresh: ${INTERVAL}s${NC}"
    
    sleep "$INTERVAL"
done

