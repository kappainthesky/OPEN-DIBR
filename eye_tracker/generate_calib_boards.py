"""
Generate two ChArUco calibration boards for Windowed Reality extrinsic calibration:

  1. PRINT BOARD  -> Main<->Helper bridging (shared board, Step 1 of your slide deck)
     Physical, printed, rigid target seen by BOTH the D455 (main) and D435 (helper)
     from different poses, at the same physical location.

  2. SCREEN BOARD  -> Helper->Screen calibration (Step 2 of your slide deck)
     Displayed full-screen on the monitor and imaged by the helper (D435) to solve
     T_helper->screen directly.

Both use ChArUco (chessboard + ArUco) so you get subpixel corner accuracy AND
robust pose even with partial occlusion / oblique viewing angles -- important
for the helper camera which will often see the screen board at a steep angle.

Different ArUco dictionaries are used for the two boards on purpose: if both
ever appear in the same frame (e.g. during debugging) there's no ID collision
or ambiguity about which board is which.

IMPORTANT: edit the CONFIG section below with your actual monitor specs and
your printer's real output before trusting the physical dimensions.
"""

import cv2
import numpy as np
import json
import os

OUT_DIR = "./calib_boards"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================== CONFIG ======================================

# ---- PRINT BOARD (main<->helper bridging) ----
PRINT_DICT      = cv2.aruco.DICT_5X5_1000
PRINT_SQUARES_X = 8
PRINT_SQUARES_Y = 6
PRINT_SQUARE_MM = 30.0          # physical square edge length after printing
PRINT_MARKER_MM = 22.0          # marker edge length (must be < square)
PRINT_DPI       = 300           # render resolution for printing

# ---- SCREEN BOARD (helper->screen) ----
SCREEN_DICT      = cv2.aruco.DICT_4X4_50
SCREEN_SQUARES_X = 10
SCREEN_SQUARES_Y = 6

# >>> EDIT THESE THREE to match YOUR monitor exactly <<<
MONITOR_DIAG_INCHES = 27.0
MONITOR_RES_X = 1920
MONITOR_RES_Y = 1080
# -----------------------------------------------------------

# =============================================================================

def px_pitch_mm(diag_in, res_x, res_y):
    """mm per pixel, from diagonal size (inches) + resolution."""
    diag_mm = diag_in * 25.4
    diag_px = (res_x**2 + res_y**2) ** 0.5
    return diag_mm / diag_px

def mm_to_px(mm, dpi):
    return int(round(mm / 25.4 * dpi))

def make_charuco_board(dict_id, squares_x, squares_y, square_len, marker_len):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, aruco_dict)
    return board, aruco_dict

def render_board(board, squares_x, squares_y, square_px, margin_px=40):
    img_w = squares_x * square_px + 2 * margin_px
    img_h = squares_y * square_px + 2 * margin_px
    img = board.generateImage((img_w, img_h), marginSize=margin_px, borderBits=1)
    return img

def add_footer(img, text, pad=60):
    """White strip below the board with ID text + a ruler mark for print verification."""
    h, w = img.shape[:2]
    canvas = np.full((h + pad, w, 3), 255, np.uint8)
    canvas[:h, :] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    cv2.putText(canvas, text, (20, h + pad - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas

results = {}

# ---------------- PRINT BOARD ----------------
print_square_px = mm_to_px(PRINT_SQUARE_MM, PRINT_DPI)
print_board, _ = make_charuco_board(PRINT_DICT, PRINT_SQUARES_X, PRINT_SQUARES_Y,
                                     PRINT_SQUARE_MM / 1000.0, PRINT_MARKER_MM / 1000.0)
print_img = render_board(print_board, PRINT_SQUARES_X, PRINT_SQUARES_Y, print_square_px)

phys_w_mm = PRINT_SQUARES_X * PRINT_SQUARE_MM
phys_h_mm = PRINT_SQUARES_Y * PRINT_SQUARE_MM
footer = (f"PRINT AT 100% / ACTUAL SIZE (no 'fit to page'). "
          f"Board should measure {phys_w_mm:.0f}mm x {phys_h_mm:.0f}mm ({phys_w_mm/25.4:.2f}in x {phys_h_mm/25.4:.2f}in) after printing. "
          f"Square = {PRINT_SQUARE_MM:.0f}mm, dict=DICT_5X5_1000")
print_img_labeled = add_footer(print_img, footer)
print_path = os.path.join(OUT_DIR, "print_board_main_helper.png")
cv2.imwrite(print_path, print_img_labeled)

results["print_board"] = {
    "role": "main<->helper bridging (shared physical board)",
    "dictionary": "DICT_5X5_1000",
    "squares_x": PRINT_SQUARES_X,
    "squares_y": PRINT_SQUARES_Y,
    "square_length_m": PRINT_SQUARE_MM / 1000.0,
    "marker_length_m": PRINT_MARKER_MM / 1000.0,
    "print_dpi": PRINT_DPI,
    "expected_physical_size_mm": [phys_w_mm, phys_h_mm],
    "file": "print_board_main_helper.png",
}

# ---------------- SCREEN BOARD ----------------
pitch = px_pitch_mm(MONITOR_DIAG_INCHES, MONITOR_RES_X, MONITOR_RES_Y)
# make squares a "nice" mm size, then convert to pixels for THIS monitor
screen_square_mm = 25.0   # target physical square size on screen
screen_square_px = int(round(screen_square_mm / pitch))
screen_marker_mm = screen_square_mm * 0.75

screen_board, _ = make_charuco_board(SCREEN_DICT, SCREEN_SQUARES_X, SCREEN_SQUARES_Y,
                                      screen_square_mm / 1000.0, screen_marker_mm / 1000.0)
screen_img = render_board(screen_board, SCREEN_SQUARES_X, SCREEN_SQUARES_Y, screen_square_px, margin_px=screen_square_px // 2)

sh, sw = screen_img.shape[:2]
if sw > MONITOR_RES_X or sh > MONITOR_RES_Y:
    print(f"WARNING: rendered board ({sw}x{sh}px) exceeds monitor resolution "
          f"({MONITOR_RES_X}x{MONITOR_RES_Y}px). Reduce SCREEN_SQUARES_X/Y or screen_square_mm.")

# place centered on a full monitor-resolution black canvas so fullscreen display = no scaling
full_canvas = np.zeros((MONITOR_RES_Y, MONITOR_RES_X, 3), np.uint8)
screen_img_bgr = cv2.cvtColor(screen_img, cv2.COLOR_GRAY2BGR) if screen_img.ndim == 2 else screen_img
y0 = max((MONITOR_RES_Y - sh) // 2, 0)
x0 = max((MONITOR_RES_X - sw) // 2, 0)
full_canvas[y0:y0+sh, x0:x0+sw] = screen_img_bgr[:min(sh, MONITOR_RES_Y - y0), :min(sw, MONITOR_RES_X - x0)]

screen_path = os.path.join(OUT_DIR, "screen_board_helper_to_screen.png")
cv2.imwrite(screen_path, full_canvas)

phys_screen_w_mm = SCREEN_SQUARES_X * screen_square_mm
phys_screen_h_mm = SCREEN_SQUARES_Y * screen_square_mm

results["screen_board"] = {
    "role": "helper->screen calibration (displayed on monitor)",
    "dictionary": "DICT_4X4_50",
    "squares_x": SCREEN_SQUARES_X,
    "squares_y": SCREEN_SQUARES_Y,
    "square_length_m": screen_square_mm / 1000.0,
    "marker_length_m": screen_marker_mm / 1000.0,
    "assumed_monitor": {
        "diagonal_inches": MONITOR_DIAG_INCHES,
        "resolution": [MONITOR_RES_X, MONITOR_RES_Y],
        "pixel_pitch_mm": round(pitch, 5),
    },
    "expected_physical_size_on_screen_mm": [phys_screen_w_mm, phys_screen_h_mm],
    "must_display_at": "100% scale, fullscreen, no OS display scaling / no image viewer zoom",
    "file": "screen_board_helper_to_screen.png",
}

with open(os.path.join(OUT_DIR, "board_specs.json"), "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"\nPixel pitch used: {pitch:.5f} mm/px")
print(f"Print board -> {print_path}  ({print_img_labeled.shape[1]}x{print_img_labeled.shape[0]}px @ {PRINT_DPI} DPI)")
print(f"Screen board -> {screen_path}  ({MONITOR_RES_X}x{MONITOR_RES_Y}px canvas)")