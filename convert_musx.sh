#!/bin/bash
# convert_musx.sh (Interactive/Manual Version)

# --- CONFIGURATION ---
WINE_PREFIX="$HOME/.wine-finale"
#FINALE_PATH="$WINE_PREFIX/drive_c/Program Files/MakeMusic/Finale/27/Finale.exe" 
FINALE_PATH="/home/sam/.wine-finale/drive_c/Program Files/MakeMusic/Finale/27/Finale.exe"
# Directories
INPUT_DIR="tools/barbershop_dataset/raw"
PROCESSED_LOG="processed_musx.log"

if [ ! -f "$FINALE_PATH" ]; then
    echo "❌ Finale.exe not found at: $FINALE_PATH"
    exit 1
fi

export WINEPREFIX="$WINE_PREFIX"
# export DISPLAY=:0
touch "$PROCESSED_LOG"

process_file() {
    local FILE="$1"
    local NAME=$(basename "$FILE")
    
    echo "🎻 Starting: $NAME"
    
    # Launch Finale with the file
    # (Optional: launch without file and use Alt+F, O sequence if preferred)
    wine "$FINALE_PATH" "$FILE" &
    PID=$!
    
    echo "   [WAIT] Please wait for Finale to fully load '$NAME'."
    echo "   [ACTION] When the score is ready and the window is active, press ENTER here..."
    echo "   (Tip: Press ctrl+f to focus this shell if needed)"
    read -r
    
    # Search for the Finale window
    WID=$(xdotool search --name "Finale" | head -1)
    if [ -z "$WID" ]; then
        echo "   ❌ Finale window not found. Skipping."
        kill $PID 2>/dev/null
        return
    fi
    xdotool windowactivate --sync $WID
    
    echo "   ...Sending export macro (Alt+F -> T -> X -> Enter)..."
    
    # 1. Open File Menu
    xdotool key --window $WID --delay 500 alt+f
    sleep 1
    
    # 2. Select Export (T)
    xdotool key --window $WID --delay 500 t
    sleep 1
    
    # 3. Select MusicXML (X)
    xdotool key --window $WID --delay 500 shift+x
    sleep 2
    
    # 4. Confirm Save Dialog (Enter)
    xdotool key --window $WID Return
    
    echo "   ...Exporting. Please wait for progress bar to finish (10s)..."
    sleep 10
    
    # 5. Close Finale
    xdotool key --window $WID alt+F4
    sleep 2
    
    # Handle "Save changes?" (No)
    xdotool key --window $WID n
    
    # Ensure process is dead before next loop
    sleep 2
    kill $PID 2>/dev/null
    
    echo "   ✅ Processed: $NAME"
    echo "$NAME" >> "$PROCESSED_LOG"
}

# --- MAIN LOOP ---
# Filter for .musx and start processing
find "$INPUT_DIR" -name "*.musx" | while read f; do
    NAME=$(basename "$f")
    if grep -q "$NAME" "$PROCESSED_LOG"; then
        echo "⏭️  Skipping $NAME (Already in log)"
    else
        process_file "$f"
        echo "------------------------------------------"
        sleep 2
    fi
done
