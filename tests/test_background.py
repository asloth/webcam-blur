"""Background image: fitting it to the frame and putting it behind the person."""

import cv2
import numpy as np


def test_fit_cover_fills_without_stretching():
    from webcam_blur.stream import fit_cover
    wide = np.zeros((100, 400, 3), np.uint8)
    wide[:, 150:250] = 255  # white centre band
    out = fit_cover(wide, 160, 120)
    assert out.shape == (120, 160, 3)
    assert out.flags["C_CONTIGUOUS"]
    # Scaled to height 120 (width 480), then the centre 160 kept: the band fills the middle.
    assert out[:, 80].min() == 255 and out[:, 0].max() == 0
    tall = fit_cover(np.zeros((900, 300, 3), np.uint8), 640, 360)
    assert tall.shape == (360, 640, 3)


def test_load_background(tmp_path):
    from webcam_blur.stream import load_background
    assert load_background(None, 64, 48) is None
    assert load_background(str(tmp_path / "missing.png"), 64, 48) is None
    path = tmp_path / "bg.png"
    cv2.imwrite(str(path), np.full((10, 20, 3), 200, np.uint8))
    bg = load_background(str(path), 64, 48)
    assert bg.shape == (48, 64, 3) and bg.dtype == np.uint8


def test_replace_background_keeps_person():
    from webcam_blur.stream import replace_background
    frame = np.full((4, 4, 3), 10, np.uint8)
    background = np.full((4, 4, 3), 250, np.uint8)
    alpha = np.zeros((4, 4), np.float32)
    alpha[:, :2] = 1.0  # person on the left half
    out = replace_background(frame, alpha, background)
    assert (out[:, :2] == 10).all() and (out[:, 2:] == 250).all()
