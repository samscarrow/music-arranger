#!/bin/bash
# convert_musx_robust.sh
#
# Usage: ./convert_musx_robust.sh
# This script wraps itself with xvfb-run. Do NOT call with xvfb-run manually.
# Opens Finale once and processes all files in a single session via AHK.

# --- CONFIGURATION ---
WINE_PREFIX="$HOME/.wine-finale"
FINALE_PATH="$WINE_PREFIX/drive_c/Program Files/MakeMusic/Finale/27/Finale.exe"
INPUT_DIR="tools/barbershop_dataset/raw"
PROCESSED_LOG="processed_musx.log"

# --- SELF-WRAP WITH XVFB ---
if [ -z "$XVFB_WRAPPED" ]; then
    export WINEPREFIX="$WINE_PREFIX"
    echo "Killing ALL wine processes..."
    pkill -9 -f wineserver 2>/dev/null
    pkill -9 -f wine 2>/dev/null
    pkill -9 -f Finale 2>/dev/null
    sleep 2
    if pgrep -f "wine" >/dev/null 2>&1; then
        echo "ERROR: Wine processes still running after kill. Aborting."
        ps aux | grep -i wine | grep -v grep
        exit 1
    fi
    echo "Clean."
    echo "Launching under xvfb-run..."
    export XVFB_WRAPPED=1
    exec xvfb-run -a --server-args="-screen 0 1024x768x24" "$0" "$@"
fi

echo "Running on DISPLAY=$DISPLAY (PID $$)"
export WINEPREFIX="$WINE_PREFIX"

trap 'echo "Caught signal, cleaning up..."; pkill -9 -f wine 2>/dev/null; exit 1' INT TERM HUP

# --- PRE-FLIGHT CHECKS ---
if [ ! -f "$FINALE_PATH" ]; then
    echo "Critical Error: Finale.exe not found at configured path."
    exit 1
fi

mkdir -p "$(dirname "$PROCESSED_LOG")"
touch "$PROCESSED_LOG"

# --- BUILD FILE LIST (skip already exported) ---
FILE_LIST="/tmp/ahk_files_$$.txt"
rm -f "$FILE_LIST"

COUNT=0
find "$INPUT_DIR" -name "*.musx" | sort | while read f; do
    EXPECTED_XML="${f%.*}.xml"
    EXPECTED_MXL="${f%.*}.mxl"
    if [ -f "$EXPECTED_XML" ] || [ -f "$EXPECTED_MXL" ]; then
        echo "Skipping $(basename "$f") (XML/MXL exists)"
    elif grep -q "$(basename "$f")" "$PROCESSED_LOG" 2>/dev/null; then
        echo "Skipping $(basename "$f") (in processed log)"
    else
        # Convert to Windows path for AHK
        ABS=$(realpath "$f")
        WIN_PATH="Z:${ABS//\//\\}"
        echo "$WIN_PATH" >> "$FILE_LIST"
    fi
done

if [ ! -f "$FILE_LIST" ] || [ ! -s "$FILE_LIST" ]; then
    echo "No files to process."
    exit 0
fi

TOTAL=$(wc -l < "$FILE_LIST")
echo "Files to process: $TOTAL"

# --- LAUNCH FINALE (no file — AHK will open each one) ---
echo "Launching Finale..."
wine "$FINALE_PATH" &>/dev/null &
WINE_PID=$!

# Wait for window
echo "Waiting for Finale window..."
WID=""
for i in {1..60}; do
    WID=$(xdotool search --name "Finale" 2>/dev/null | head -1)
    if [ -n "$WID" ]; then
        break
    fi
    sleep 1
done

if [ -z "$WID" ]; then
    echo "Timeout: Finale window never appeared."
    pkill -9 -f wine 2>/dev/null
    exit 1
fi
echo "Finale window found after ${i}s"

# --- LAUNCH AHK BATCH EXPORT ---
AHK_EXE="C:\\tools\\AHK\\AutoHotkey64.exe"
AHK_SCRIPT="Z:$(realpath "$(dirname "$0")/export_musicxml.ahk" | sed 's|/|\\|g')"
SIGNAL_FILE="/tmp/ahk_signal_$$"
SIGNAL_WIN="Z:${SIGNAL_FILE//\//\\}"
FILE_LIST_WIN="Z:${FILE_LIST//\//\\}"
AHK_LOG="/tmp/ahk_log_$$.txt"
AHK_LOG_WIN="Z:${AHK_LOG//\//\\}"

rm -f "$SIGNAL_FILE" "$AHK_LOG"

echo "Launching AHK batch export..."
wine "$AHK_EXE" "$AHK_SCRIPT" "$FILE_LIST_WIN" "$SIGNAL_WIN" "$AHK_LOG_WIN" &>/dev/null &

# --- MONITOR PROGRESS ---
LAST_LINES=0
TIMEOUT=0
MAX_TIMEOUT=600  # 10 minutes total max

while [ "$TIMEOUT" -lt "$MAX_TIMEOUT" ]; do
    if [ -f "$SIGNAL_FILE" ]; then
        CURRENT_LINES=$(wc -l < "$SIGNAL_FILE")
        if [ "$CURRENT_LINES" -gt "$LAST_LINES" ]; then
            # Print new lines
            tail -n +$((LAST_LINES + 1)) "$SIGNAL_FILE" | while read line; do
                echo "   $line"

                # Check for completion
                if [ "$line" = "ALL_DONE" ]; then
                    break 2
                fi

                # Record successes
                if [[ "$line" == OK:* ]]; then
                    FNAME="${line#OK:}"
                    echo "$FNAME" >> "$PROCESSED_LOG"
                fi
            done
            LAST_LINES=$CURRENT_LINES
            TIMEOUT=0  # reset timeout on progress
        fi

        # Check if ALL_DONE appeared
        if grep -q "ALL_DONE" "$SIGNAL_FILE" 2>/dev/null; then
            break
        fi
    fi
    sleep 1
    TIMEOUT=$((TIMEOUT + 1))
done

if [ "$TIMEOUT" -ge "$MAX_TIMEOUT" ]; then
    echo "Timeout: no progress for ${MAX_TIMEOUT}s"
fi

# --- PRINT SUMMARY ---
echo ""
echo "=== RESULTS ==="
if [ -f "$SIGNAL_FILE" ]; then
    OK_COUNT=$(grep -c "^OK:" "$SIGNAL_FILE" 2>/dev/null || echo 0)
    FAIL_COUNT=$(grep -c "^FAIL:" "$SIGNAL_FILE" 2>/dev/null || echo 0)
    echo "Succeeded: $OK_COUNT"
    echo "Failed:    $FAIL_COUNT"
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "Failures:"
        grep "^FAIL:" "$SIGNAL_FILE" | sed 's/^/  /'
    fi
fi

# Print AHK log
if [ -f "$AHK_LOG" ]; then
    echo ""
    echo "--- AHK log ---"
    cat "$AHK_LOG" | sed 's/^/   /'
    echo "--- end AHK log ---"
fi

# --- VERIFY EXPORTED FILES ---
echo ""
echo "=== VERIFICATION ==="
EXPORTED=0
MISSING=0
while read WIN_PATH; do
    # Convert back to Linux path (Z:\home\... -> /home/...)
    LINUX_PATH=$(echo "$WIN_PATH" | sed 's|\\|/|g; s|^Z:||')
    BASE="${LINUX_PATH%.*}"
    FNAME=$(basename "$LINUX_PATH")
    if [ -f "${BASE}.mxl" ] || [ -f "${BASE}.xml" ]; then
        EXPORTED=$((EXPORTED + 1))
    else
        echo "  MISSING: $FNAME"
        MISSING=$((MISSING + 1))
    fi
done < "$FILE_LIST"
echo "Exported: $EXPORTED / $TOTAL"
[ "$MISSING" -gt 0 ] && echo "Missing:  $MISSING"

# --- CLEANUP ---
pkill -9 -f wine 2>/dev/null
rm -f "$SIGNAL_FILE" "$FILE_LIST" "$AHK_LOG"
echo "Done."
