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


def test_fit_cover_keeps_proportions():
    from webcam_blur.stream import fit_cover
    # A centred circle stays a circle: stretching a 400x100 image to 160x120 would flatten it.
    image = np.zeros((100, 400, 3), np.uint8)
    cv2.circle(image, (200, 50), 40, (255, 255, 255), -1)
    out = fit_cover(image, 160, 120)
    ys, xs = np.nonzero(out[:, :, 0] > 127)
    width, height = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    assert abs(width / height - 1) < 0.05
    # Downscaling path too (INTER_AREA): a 2000x1000 image into 640x360.
    big = np.zeros((1000, 2000, 3), np.uint8)
    cv2.circle(big, (1000, 500), 300, (255, 255, 255), -1)
    out = fit_cover(big, 640, 360)
    ys, xs = np.nonzero(out[:, :, 0] > 127)
    assert abs((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1) - 1) < 0.05


def test_fit_cover_narrow_image_stays_small(monkeypatch):
    from webcam_blur import stream
    sizes = []
    real_resize = cv2.resize

    def spy(src, dsize, **kw):
        sizes.append(dsize[0] * dsize[1])
        return real_resize(src, dsize, **kw)

    monkeypatch.setattr(stream.cv2, "resize", spy)
    out = stream.fit_cover(np.zeros((4000, 100, 3), np.uint8), 1920, 1080)
    assert out.shape == (1080, 1920, 3)
    assert max(sizes) <= 1920 * 1080  # resized straight to the frame, never a huge intermediate


def _wait(loader, path, timeout=5.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = loader.get(path)
        if loader.loading is None and (out is not None or loader.shown_path == path):
            return out
        time.sleep(0.01)
    raise AssertionError("background did not load")


def test_loader_retries_a_file_that_appears_later(tmp_path):
    from webcam_blur.stream import BackgroundLoader
    path = str(tmp_path / "late.png")
    loader = BackgroundLoader(32, 24)
    assert _wait(loader, path) is None  # missing: blur
    cv2.imwrite(path, np.full((10, 10, 3), 200, np.uint8))
    loader.next_try = 0.0  # skip the RETRY wait
    out = _wait(loader, path)
    assert out is not None and out.shape == (24, 32, 3)


def test_loader_keeps_previous_image_while_switching_and_clears_for_blur(tmp_path):
    from webcam_blur.stream import BackgroundLoader
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    cv2.imwrite(a, np.full((10, 10, 3), 50, np.uint8))
    cv2.imwrite(b, np.full((10, 10, 3), 150, np.uint8))
    loader = BackgroundLoader(8, 8)
    assert _wait(loader, a)[0, 0, 0] == 50
    first = loader.get(b)  # b starts loading: a stays on screen meanwhile
    assert first is not None and first[0, 0, 0] in (50, 150)
    assert _wait(loader, b)[0, 0, 0] == 150
    assert loader.get(None) is None
