#!/bin/bash

# This script monitors the process with the highest memory footprint on the system.
# Ensure that the pre-trained model parameters are already sliced before running this script
# to get an accurate measurement of the model inference memory footprint.

# Initialize variables to track maximum observed memory usage
MAX_PHYSICAL=0            # Maximum RSS observed
MAX_PEAK=0                # Maximum peak RSS observed (VmHWM)
MAX_PID=0                 # PID of the process with maximum memory usage
MAX_CMD=""                # Command of the process with maximum memory usage
MAX_TIMESTAMP=""          # Timestamp when the maximum memory usage was observed

# Initialize variables to track current memory usage
CURRENT_PHYSICAL=0        # Current RSS of the process
CURRENT_PEAK=0            # Current peak RSS of the process (VmHWM)
CURRENT_CMD=""            # Command of the current process
CURRENT_PID=0             # PID of the current process

# Function to display current and highest memory usage details
display_info() {
    echo -e "\033[H\033[J"  # Clear the screen without flashing
    echo "Monitoring highest memory usage process... (Press 'q' to quit)"
    echo "--------------------------------------------------------------------"
    echo "Timestamp: $MAX_TIMESTAMP"
    echo "Process PID: $MAX_PID"
    echo "Command: $MAX_CMD"
    echo "Highest Memory Usage:"
    echo "  - Physical Footprint: $(bc <<< "scale=2; $MAX_PHYSICAL/1024/1024") GB"
    echo "  - Physical Footprint (Peak): $(bc <<< "scale=2; $MAX_PEAK/1024/1024") GB"
    echo "--------------------------------------------------------------------"
    echo "Current Memory Usage:"
    echo "  - Current Process with Maximum Memory Usage: $CURRENT_CMD"
    echo "  - Current Process PID: $CURRENT_PID"
    echo "  - Current Physical Footprint: $(bc <<< "scale=2; $CURRENT_PHYSICAL/1024/1024") GB"
    echo "  - Current Physical Footprint (Peak): $(bc <<< "scale=2; $CURRENT_PEAK/1024/1024") GB"
    echo "--------------------------------------------------------------------"
}

# Function to convert memory sizes from strings (e.g., "2.1G") into kilobytes
convert_to_kb() {
    local size=$(echo "$1" | tr -d '[:alpha:],')  # Remove non-numeric characters
    if [[ $1 == *G || $1 == *g ]]; then
        echo "$(bc <<< "$size * 1024 * 1024")"
    elif [[ $1 == *M || $1 == *m ]]; then
        echo "$(bc <<< "$size * 1024")"
    elif [[ $1 == *K || $1 == *k ]]; then
        echo "$size"
    else
        echo "0"  # Return 0 if conversion fails
    fi
}

# Capture Ctrl + C and 'q' key to stop monitoring and exit
trap 'echo -e "\nMonitoring stopped."; exit 0' SIGINT

# Main monitoring loop
while true; do
    # Check for 'q' key press to exit with a 1-second timeout
    read -t 1 -n 1 key
    if [[ $key == "q" ]]; then
        echo -e "\nMonitoring stopped by user."
        exit 0
    fi

    # Get the PID of the process with the highest memory usage
    PID=$(ps axo pid,rss | awk 'NR>1 {print $1, $2}' | sort -k2 -n | tail -n 1 | awk '{print $1}')

    # Skip iteration if no valid PID is found
    if [[ -z "$PID" || "$PID" == "0" ]]; then
        sleep 0.5
        continue
    fi

    # Use vmmap to capture memory statistics of the current process
    VMMAP_OUTPUT=$(vmmap $PID 2>/dev/null)

    # Extract the current physical footprint and peak physical footprint
    PHYSICAL=$(echo "$VMMAP_OUTPUT" | grep -i "Physical footprint:" | head -n 1 | awk '{print $3}')
    PEAK=$(echo "$VMMAP_OUTPUT" | grep -i "Physical footprint (peak):" | head -n 1 | awk '{print $4}')

    # Convert extracted values to kilobytes
    PHYSICAL=$(convert_to_kb "$PHYSICAL")
    PEAK=$(convert_to_kb "$PEAK")

    # Skip iteration if parsing failed
    if [[ -z "$PHYSICAL" || -z "$PEAK" ]]; then
        sleep 0.5
        continue
    fi

    # Update current process details
    CURRENT_CMD=$(ps -p $PID -o command=)
    CURRENT_PHYSICAL=$PHYSICAL
    CURRENT_PEAK=$PEAK
    CURRENT_PID=$PID

    # Get the current timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Update maximum values if the current values exceed them
    if (( $(bc <<< "$PHYSICAL > $MAX_PHYSICAL") == 1 )); then
        MAX_PHYSICAL=$PHYSICAL
        MAX_TIMESTAMP=$TIMESTAMP
        MAX_PID=$PID
        MAX_CMD=$CURRENT_CMD
    fi
    if (( $(bc <<< "$PEAK > $MAX_PEAK") == 1 )); then
        MAX_PEAK=$PEAK
    fi

    # Display current and highest memory usage information
    display_info

    # Refresh every 0.5 seconds
    sleep 0.5
done
