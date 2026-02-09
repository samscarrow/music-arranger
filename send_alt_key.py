#!/usr/bin/env python3
"""Send Alt+key to a Wine window using XSendEvent with modifier in event state.

Usage: python3 send_alt_key.py <window_id> <key_char>
Example: python3 send_alt_key.py 14680065 f
"""
import sys
from Xlib import X, XK, display, protocol

def send_alt_key(wid, key_char):
    d = display.Display()
    window = d.create_resource_object('window', wid)

    # Get keycode for the character
    keysym = XK.string_to_keysym(key_char)
    keycode = d.keysym_to_keycode(keysym)
    if not keycode:
        print(f"No keycode for '{key_char}'")
        sys.exit(1)

    print(f"Sending Alt+{key_char} (keycode={keycode}) to window {wid}")

    # Mod1Mask = Alt modifier
    mod_mask = X.Mod1Mask

    # Send KeyPress with Alt in state
    event = protocol.event.KeyPress(
        time=X.CurrentTime,
        root=d.screen().root,
        window=window,
        child=X.NONE,
        root_x=0, root_y=0,
        event_x=0, event_y=0,
        state=mod_mask,
        detail=keycode,
        same_screen=1,
    )
    window.send_event(event, event_mask=X.KeyPressMask)

    # Send KeyRelease with Alt in state
    event = protocol.event.KeyRelease(
        time=X.CurrentTime,
        root=d.screen().root,
        window=window,
        child=X.NONE,
        root_x=0, root_y=0,
        event_x=0, event_y=0,
        state=mod_mask,
        detail=keycode,
        same_screen=1,
    )
    window.send_event(event, event_mask=X.KeyReleaseMask)

    d.flush()
    print("Sent.")

if __name__ == '__main__':
    wid = int(sys.argv[1])
    key = sys.argv[2]
    send_alt_key(wid, key)
