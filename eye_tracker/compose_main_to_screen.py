"""
Compose T_main->helper and T_helper->screen into the final T_main->screen.

Step 3 of your slide deck: chain the two extrinsic calibrations together so
you have a single main-camera-to-screen transform, without ever needing main
to directly see the screen board.

    T_main->screen = T_main->helper @ T_helper->screen

i.e. for a point X expressed in screen-board coordinates:
    X_helper = R_helper_screen @ X_screen + t_helper_screen
    X_main   = R_main_helper   @ X_helper + t_main_helper
             = (R_main_helper @ R_helper_screen) @ X_screen
               + (R_main_helper @ t_helper_screen + t_main_helper)

Inputs (must already exist, produced by your earlier two calibration steps):
    main_to_helper_calib.json   (R, t = T_main->helper)
    helper_to_screen_calib.json (R, t = T_helper->screen,
                                  plus screen_center_helper_cam)

Output:
    main_to_screen_calib.json containing:
        - R (3x3), t (3x1)            -> T_main->screen as [R|t]
        - screen_center_main_cam      -> screen center point, in main-cam frame
        - distance_main_to_screen_center_mm -> quick physical sanity-check number
                                          (measure this with a tape measure
                                          against your real rig)
"""

import json
import numpy as np

MAIN_TO_HELPER_FILE   = "main_to_helper_calib.json"
HELPER_TO_SCREEN_FILE = "helper_to_screen_calib.json"
OUTPUT_FILE           = "main_to_screen_calib.json"


def load_transform(path):
    with open(path, "r") as f:
        data = json.load(f)
    R = np.array(data["R"])
    t = np.array(data["t"])
    return R, t, data


def check_rotation(R, name):
    orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
    det = np.linalg.det(R)
    ok = orth_err < 1e-6 and abs(det - 1.0) < 1e-6
    status = "OK" if ok else "WARNING -- not a clean rotation matrix"
    print(f"{name}: orthogonality error = {orth_err:.2e}, det = {det:.6f}  [{status}]")
    return ok


def main():
    R_mh, t_mh, _ = load_transform(MAIN_TO_HELPER_FILE)
    R_hs, t_hs, helper_screen_data = load_transform(HELPER_TO_SCREEN_FILE)

    ok1 = check_rotation(R_mh, "R_main_helper")
    ok2 = check_rotation(R_hs, "R_helper_screen")
    if not (ok1 and ok2):
        print("\nAborting -- one of the input rotation matrices failed validation. "
              "Re-check the source calibration files before composing.")
        return

    R_ms = R_mh @ R_hs
    t_ms = R_mh @ t_hs + t_mh

    result = {
        "R": R_ms.tolist(),
        "t": t_ms.tolist(),
    }

    # Carry the screen-center point through too, if the helper->screen file has it --
    # it's a convenient single number to sanity-check with a tape measure.
    if "screen_center_helper_cam" in helper_screen_data:
        screen_center_helper = np.array(helper_screen_data["screen_center_helper_cam"])
        screen_center_main = R_mh @ screen_center_helper + t_mh
        result["screen_center_main_cam"] = screen_center_main.tolist()
        result["distance_main_to_screen_center_mm"] = float(np.linalg.norm(screen_center_main) * 1000)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {OUTPUT_FILE}")
    print(f"R_main_screen:\n{R_ms}")
    print(f"t_main_screen (m): {t_ms}")
    if "distance_main_to_screen_center_mm" in result:
        print(f"\ndistance from main camera to screen center: "
              f"{result['distance_main_to_screen_center_mm']:.1f} mm")
        print("Sanity check: measure this with a tape measure on your physical rig. "
              "Should be in the same ballpark -- large disagreement (2x or more) "
              "points to a mismatch upstream (board dimensions, resolution mismatch "
              "between calibration and runtime, or a swapped axis).")


if __name__ == "__main__":
    main()
