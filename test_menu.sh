#!/bin/bash
# Test: click at exact menu bar position (top edge of window)
WINE_PREFIX="$HOME/.wine-finale"
FINALE_PATH="$WINE_PREFIX/drive_c/Program Files/MakeMusic/Finale/27/Finale.exe"
TEST_FILE="tools/barbershop_dataset/raw/16_Going_On_17_3682.musx"

if [ -z "$XVFB_WRAPPED" ]; then
    export WINEPREFIX="$WINE_PREFIX"
    pkill -9 -f wineserver 2>/dev/null
    pkill -9 -f wine 2>/dev/null
    sleep 2
    pgrep -f "wine" >/dev/null 2>&1 && { echo "ERROR"; exit 1; }
    export XVFB_WRAPPED=1
    exec xvfb-run -a --server-args="-screen 0 1024x768x24" "$0" "$@"
fi

export WINEPREFIX="$WINE_PREFIX"
trap 'pkill -9 -f wine 2>/dev/null; exit 1' INT TERM HUP
echo "DISPLAY=$DISPLAY"

dismiss_window() {
    local name="$1"; shift
    local wid=$(xdotool search --name "$name" 2>/dev/null | head -1)
    if [ -n "$wid" ]; then
        echo "  Dismissing '$name'"
        xdotool windowfocus --sync $wid 2>/dev/null; sleep 0.3
        for key in "$@"; do xdotool key "$key"; sleep 0.3; done
        sleep 1; return 0
    fi
    return 1
}

take_screenshot() {
    xwd -root -out "/tmp/dialog_${1}.xwd" 2>/dev/null && \
    magick "/tmp/dialog_${1}.xwd" "/tmp/dialog_${1}.png" 2>/dev/null
    echo "[screenshot: /tmp/dialog_${1}.png]"
}

wine "$FINALE_PATH" "$TEST_FILE" &>/dev/null &
WID=""
for i in {1..60}; do
    WID=$(xdotool search --name "Finale" 2>/dev/null | head -1)
    [ -n "$WID" ] && break
    sleep 1
done
[ -z "$WID" ] && { echo "Timeout!"; pkill -9 -f wine 2>/dev/null; exit 1; }
echo "Window $WID after ${i}s"
sleep 5

for attempt in {1..10}; do
    dismiss_window "Finale Authorization Wizard" Tab Tab space && continue
    dismiss_window "Authorize Finale" Tab Tab space && continue
    dismiss_window "Default MIDI" Return && continue
    DISMISSED=0
    for wid in $(xdotool search --name "Finale" 2>/dev/null); do
        eval $(xdotool getwindowgeometry --shell $wid 2>/dev/null)
        if [ "${WIDTH:-0}" -lt 500 ] && [ "${HEIGHT:-0}" -lt 500 ]; then
            xdotool windowfocus --sync $wid 2>/dev/null; sleep 0.3
            xdotool key Return; sleep 1; DISMISSED=1; break
        fi
    done
    [ $DISMISSED -eq 1 ] && continue
    break
done

eval $(xdotool getwindowgeometry --shell $WID 2>/dev/null)
echo "Window at ($X, $Y) ${WIDTH}x${HEIGHT}"

# Click body for focus
xdotool mousemove --sync $((X + WIDTH/2)) $((Y + HEIGHT/2))
sleep 0.2; xdotool click 1; sleep 1

# Try every pixel from y=26 to y=29 (very top of window)
for test_y in 26 27 28 29; do
    echo "=== Click at (14, $test_y) ==="
    xdotool mousemove --sync 14 $test_y
    sleep 0.3
    xdotool click 1
    sleep 1.5
    take_screenshot "click_14_${test_y}"
    xdotool key Escape; sleep 0.5
done

pkill -9 -f wine 2>/dev/null
sleep 2
