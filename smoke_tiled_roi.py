#!/usr/bin/env python3
import argparse

import backend


def _run_real(image_path):
    params = backend.build_detection_params_for_preset("legacy")
    params["detector_backend"] = "stardist"
    backend.set_detection_params(params)

    nuclei_full, report_full = backend.detect_nuclei_tiled(
        image_path=image_path,
        average_cell_diameter_px=12.0,
        tile_factor=64.0,
        overlap_ratio=0.25,
        tile_min_px=512,
        tile_max_px=2048,
        border_trim_enabled=True,
        border_trim_factor=1.0,
    )

    image = backend.load_image(image_path)
    h, w = image.shape[:2]
    roi = [0.0, 0.0, float(w) * 0.5, float(h) * 0.5]
    nuclei_roi, report_roi = backend.detect_nuclei_tiled(
        image_path=image_path,
        average_cell_diameter_px=12.0,
        tile_factor=64.0,
        overlap_ratio=0.25,
        tile_min_px=512,
        tile_max_px=2048,
        roi_rect=[roi],
        border_trim_enabled=True,
        border_trim_factor=1.0,
    )

    x, y, rw, rh = roi
    for nucleus in nuclei_roi:
        cx, cy = nucleus.get("center", (None, None))
        if cx is None or cy is None:
            continue
        if not (x <= float(cx) <= x + rw and y <= float(cy) <= y + rh):
            raise AssertionError("Found nucleus center outside ROI in real run")

    print(f"SMOKE_REAL_OK full_count={len(nuclei_full)} roi_count={len(nuclei_roi)}")
    print(f"  full_tiles_used={report_full.get('tiles_used', 0)} roi_tiles_used={report_roi.get('tiles_used', 0)}")


def _run_mock(image_path):
    image = backend.load_image(image_path)
    h, w = image.shape[:2]

    original_detect = backend.detect_nuclei_in_image

    def fake_detect(tile_image, enhancement_params=None):
        th, tw = tile_image.shape[:2]
        return [
            {
                "center": (float(tw) * 0.25, float(th) * 0.25),
                "contour": [
                    (float(tw) * 0.20, float(th) * 0.20),
                    (float(tw) * 0.30, float(th) * 0.20),
                    (float(tw) * 0.30, float(th) * 0.30),
                    (float(tw) * 0.20, float(th) * 0.30),
                ],
                "area_px": float(max(4.0, tw * th * 0.005)),
            },
            {
                "center": (float(tw) * 0.85, float(th) * 0.85),
                "contour": [
                    (float(tw) * 0.80, float(th) * 0.80),
                    (float(tw) * 0.90, float(th) * 0.80),
                    (float(tw) * 0.90, float(th) * 0.90),
                    (float(tw) * 0.80, float(th) * 0.90),
                ],
                "area_px": float(max(4.0, tw * th * 0.006)),
            },
        ]

    backend.detect_nuclei_in_image = fake_detect
    try:
        nuclei_full, report_full = backend.detect_nuclei_tiled_in_image(
            image_bgr=image,
            average_cell_diameter_px=12.0,
            tile_factor=64.0,
            overlap_ratio=0.25,
            border_trim_enabled=False,
            merge_iou_thresh=0.3,
        )

        roi = [0.0, 0.0, float(w) * 0.5, float(h) * 0.5]
        nuclei_roi, report_roi = backend.detect_nuclei_tiled_in_image(
            image_bgr=image,
            average_cell_diameter_px=12.0,
            tile_factor=64.0,
            overlap_ratio=0.25,
            roi_rect=[roi],
            border_trim_enabled=False,
            merge_iou_thresh=0.3,
        )
    finally:
        backend.detect_nuclei_in_image = original_detect

    x, y, rw, rh = roi
    for nucleus in nuclei_roi:
        cx, cy = nucleus.get("center", (None, None))
        if cx is None or cy is None:
            continue
        if not (x <= float(cx) <= x + rw and y <= float(cy) <= y + rh):
            raise AssertionError("Found nucleus center outside ROI in mock run")

    print(f"SMOKE_MOCK_OK full_count={len(nuclei_full)} roi_count={len(nuclei_roi)}")
    print(f"  full_tiles_used={report_full.get('tiles_used', 0)} roi_tiles_used={report_roi.get('tiles_used', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Smoke checks for tiled inference and ROI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run smoke with mocked tile detector (works without StarDist/Cellpose packages)",
    )
    args = parser.parse_args()

    if args.mock:
        _run_mock(args.image)
        return

    _run_real(args.image)


if __name__ == "__main__":
    main()
