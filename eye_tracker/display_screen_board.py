"""
Display the screen ChArUco board fullscreen on a SPECIFIC monitor, leaving
your other monitor free to run code/terminal on.

Usage:
    python display_screen_board.py

It will list detected monitors with their resolution and position, then ask
which one to show the board on. The window is moved to that monitor's exact
pixel offset before going fullscreen, so it lands on the right screen instead
of wherever the window manager defaults to.

Controls:
    q or ESC  -> quit
"""

import cv2
import subprocess
import re
import sys

IMG_PATH = "calib_boards/screen_board_helper_to_screen.png"  # adjust to your path
WINDOW_NAME = "screen_board"


def get_monitors():
    """Parse `xrandr` output for connected monitors: name, width, height, x_offset, y_offset."""
    try:
        out = subprocess.check_output(["xrandr", "--query"], text=True)
    except Exception as e:
        print(f"Could not run xrandr ({e}). Falling back to single-monitor fullscreen.")
        return []

    monitors = []
    # lines like: "DP-1 connected 1920x1080+1920+0 ..."
    pattern = re.compile(r"^(\S+) connected.*?(\d+)x(\d+)\+(\d+)\+(\d+)")
    for line in out.splitlines():
        m = pattern.match(line)
        if m:
            name, w, h, x, y = m.groups()
            monitors.append({
                "name": name,
                "width": int(w),
                "height": int(h),
                "x": int(x),
                "y": int(y),
            })
    return monitors


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"ERROR: could not load {IMG_PATH}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"Loaded board: {w}x{h}px\n")

    monitors = get_monitors()

    target_x, target_y = 0, 0

    if monitors:
        print("Detected monitors:")
        for i, mon in enumerate(monitors):
            print(f"  [{i}] {mon['name']}: {mon['width']}x{mon['height']} at offset ({mon['x']}, {mon['y']})")

        choice = input(f"\nWhich monitor should show the board? [0-{len(monitors)-1}]: ").strip()
        try:
            idx = int(choice)
            mon = monitors[idx]
            target_x, target_y = mon["x"], mon["y"]
            if (mon["width"], mon["height"]) != (w, h):
                print(f"WARNING: board is {w}x{h}px but chosen monitor is "
                      f"{mon['width']}x{mon['height']}px. Regenerate the board "
                      f"at this monitor's resolution for exact 1:1 scale, or "
                      f"the physical square size will be off.")
        except (ValueError, IndexError):
            print("Invalid choice, defaulting to (0,0).")
    else:
        print("No monitor list available -- window will open at (0,0). "
              "Manually drag it to the correct screen before pressing fullscreen if needed.")

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(WINDOW_NAME, target_x, target_y)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(WINDOW_NAME, img)

    print("\nBoard should now be fullscreen on the chosen monitor.")
    print("Switch focus to your other monitor to run your capture script.")
    print("Press 'q' or ESC in this window (click it first) to close when done.")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()