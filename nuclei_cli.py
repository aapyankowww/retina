#!/usr/bin/env python3
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import backend


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return data


def _parse_float_grid(raw):
    values = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("Empty numeric grid")
    return values


def _parse_int_grid(raw):
    values = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(int(float(text)))
    if not values:
        raise ValueError("Empty integer grid")
    return values


def _parse_bool_grid(raw):
    values = []
    for chunk in str(raw).split(","):
        text = chunk.strip().lower()
        if not text:
            continue
        if text in {"1", "true", "yes", "y", "on"}:
            values.append(True)
        elif text in {"0", "false", "no", "n", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Cannot parse bool value: {chunk}")
    if not values:
        raise ValueError("Empty bool grid")
    return values


def _parse_rect_string(raw):
    parts = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    if len(parts) != 4:
        raise ValueError(f"Rect must have 4 values x,y,w,h: {raw}")
    return [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])]


def _parse_poly_string(raw):
    parts = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    if len(parts) < 6 or len(parts) % 2 != 0:
        raise ValueError(f"Polygon must have even count of values x1,y1,x2,y2,...: {raw}")

    points = []
    for idx in range(0, len(parts), 2):
        points.append([float(parts[idx]), float(parts[idx + 1])])
    return points


def _coerce_rect_entries(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "rect" in raw:
            return _coerce_rect_entries(raw["rect"])
        if "rects" in raw:
            return _coerce_rect_entries(raw["rects"])
        return []
    if isinstance(raw, str):
        return [_parse_rect_string(raw)]
    if isinstance(raw, (list, tuple)):
        if len(raw) == 4 and isinstance(raw[0], (int, float)):
            return [[float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])]]
        rects = []
        for item in raw:
            rects.extend(_coerce_rect_entries(item))
        return rects
    return []


def _coerce_poly_entries(raw):
    if raw is None:
        return []
    if isinstance(raw, dict):
        if "polygon" in raw:
            return _coerce_poly_entries(raw["polygon"])
        if "polygons" in raw:
            return _coerce_poly_entries(raw["polygons"])
        return []
    if isinstance(raw, str):
        return [_parse_poly_string(raw)]
    if isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], (list, tuple)):
            first = raw[0]
            if len(first) >= 2 and isinstance(first[0], (int, float)):
                points = []
                for p in raw:
                    if len(p) < 2:
                        continue
                    points.append([float(p[0]), float(p[1])])
                if len(points) >= 3:
                    return [points]
        polys = []
        for item in raw:
            polys.extend(_coerce_poly_entries(item))
        return polys
    return []


def _append_geojson_geometry(geometry, rects, polys):
    if not isinstance(geometry, dict):
        return
    gtype = str(geometry.get("type", "")).strip().lower()
    coords = geometry.get("coordinates")

    if gtype == "polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
        poly = _coerce_poly_entries(ring)
        polys.extend(poly)
        return

    if gtype == "multipolygon" and isinstance(coords, list):
        for poly in coords:
            if not poly:
                continue
            ring = poly[0]
            polys.extend(_coerce_poly_entries(ring))
        return

    if gtype == "feature":
        _append_geojson_geometry(geometry.get("geometry"), rects, polys)
        return

    if gtype == "featurecollection":
        for feature in geometry.get("features", []):
            _append_geojson_geometry(feature, rects, polys)


def _load_rois_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rects = []
    polys = []

    def walk(obj):
        if obj is None:
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return
        if isinstance(obj, dict):
            otype = str(obj.get("type", "")).strip().lower()
            if otype in {"polygon", "multipolygon", "feature", "featurecollection"}:
                _append_geojson_geometry(obj, rects, polys)
                return

            if "rect" in obj:
                rects.extend(_coerce_rect_entries(obj["rect"]))
            if "rects" in obj:
                rects.extend(_coerce_rect_entries(obj["rects"]))
            if "polygon" in obj:
                polys.extend(_coerce_poly_entries(obj["polygon"]))
            if "polygons" in obj:
                polys.extend(_coerce_poly_entries(obj["polygons"]))

            for value in obj.values():
                if isinstance(value, (dict, list)):
                    walk(value)

    walk(data)
    return rects, polys


def _collect_overrides(args):
    overrides = {}

    if args.detector_backend:
        overrides["detector_backend"] = args.detector_backend
    if args.model_name:
        overrides["model_name"] = args.model_name

    for key in [
        "prob_thresh",
        "nms_thresh",
        "scale",
        "min_area_px",
        "max_area_px",
        "min_purple_ratio",
        "purple_h_min",
        "purple_h_max",
        "purple_s_min",
        "purple_v_max",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    if getattr(args, "disable_purple_filter", False):
        overrides["purple_filter_enabled"] = False
    if getattr(args, "enable_purple_filter", False):
        overrides["purple_filter_enabled"] = True

    if getattr(args, "require_center_purple", False):
        overrides["require_center_purple"] = True
    if getattr(args, "no_require_center_purple", False):
        overrides["require_center_purple"] = False

    if getattr(args, "strict_purple_filter", False):
        overrides["strict_purple_filter"] = True
    if getattr(args, "soft_purple_filter", False):
        overrides["strict_purple_filter"] = False

    return overrides


def _collect_tiling_options(args, config_data):
    opts = {
        "sliced_inference": bool(config_data.get("sliced_inference", False)),
        "average_cell_diameter_px": config_data.get("average_cell_diameter_px"),
        "average_cell_bbox": config_data.get("average_cell_bbox"),
        "tile_factor": float(config_data.get("tile_factor", 64.0)),
        "overlap_ratio": float(config_data.get("overlap_ratio", 0.25)),
        "tile_min_px": int(config_data.get("tile_min_px", 512)),
        "tile_max_px": int(config_data.get("tile_max_px", 2048)),
        "min_roi_cover_ratio": float(config_data.get("min_roi_cover_ratio", 0.05)),
        "merge_iou_thresh": float(config_data.get("merge_iou_thresh", 0.30)),
        "border_trim_enabled": bool(config_data.get("border_trim_enabled", True)),
        "border_trim_factor": float(config_data.get("border_trim_factor", 1.0)),
        "roi_rect": [],
        "roi_poly": [],
    }

    if isinstance(opts["average_cell_bbox"], str) and opts["average_cell_bbox"].strip():
        opts["average_cell_bbox"] = _parse_rect_string(opts["average_cell_bbox"])

    if config_data.get("roi_rect") is not None:
        opts["roi_rect"].extend(_coerce_rect_entries(config_data.get("roi_rect")))
    if config_data.get("roi_poly") is not None:
        opts["roi_poly"].extend(_coerce_poly_entries(config_data.get("roi_poly")))

    cfg_roi_file = str(config_data.get("roi_file", "") or "").strip()
    if cfg_roi_file:
        file_rects, file_polys = _load_rois_from_file(cfg_roi_file)
        opts["roi_rect"].extend(file_rects)
        opts["roi_poly"].extend(file_polys)

    if args.sliced_inference:
        opts["sliced_inference"] = True

    if args.average_cell_diameter_px is not None:
        opts["average_cell_diameter_px"] = float(args.average_cell_diameter_px)
    if args.average_cell_bbox:
        opts["average_cell_bbox"] = _parse_rect_string(args.average_cell_bbox)

    if args.tile_factor is not None:
        opts["tile_factor"] = float(args.tile_factor)
    if args.overlap_ratio is not None:
        opts["overlap_ratio"] = float(args.overlap_ratio)
    if args.tile_min_px is not None:
        opts["tile_min_px"] = int(args.tile_min_px)
    if args.tile_max_px is not None:
        opts["tile_max_px"] = int(args.tile_max_px)
    if args.min_roi_cover_ratio is not None:
        opts["min_roi_cover_ratio"] = float(args.min_roi_cover_ratio)
    if args.merge_iou_thresh is not None:
        opts["merge_iou_thresh"] = float(args.merge_iou_thresh)

    if args.border_trim_enabled:
        opts["border_trim_enabled"] = True
    if args.border_trim_disabled:
        opts["border_trim_enabled"] = False
    if args.border_trim_factor is not None:
        opts["border_trim_factor"] = float(args.border_trim_factor)

    if args.roi_rect:
        opts["roi_rect"].extend([_parse_rect_string(x) for x in args.roi_rect])
    if args.roi_poly:
        opts["roi_poly"].extend([_parse_poly_string(x) for x in args.roi_poly])

    if args.roi_file:
        file_rects, file_polys = _load_rois_from_file(args.roi_file)
        opts["roi_rect"].extend(file_rects)
        opts["roi_poly"].extend(file_polys)

    return opts


def _area_stats(nuclei, small_area_px, large_area_px):
    if not nuclei:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "pct_small": 0.0,
            "pct_large": 0.0,
        }

    areas = np.asarray([float(n.get("area_px", 0.0)) for n in nuclei], dtype=np.float32)
    if areas.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "pct_small": 0.0,
            "pct_large": 0.0,
        }

    small_thr = float(max(0.0, small_area_px))
    large_thr = float(max(small_thr, large_area_px))
    return {
        "count": int(areas.size),
        "min": float(np.min(areas)),
        "median": float(np.median(areas)),
        "mean": float(np.mean(areas)),
        "max": float(np.max(areas)),
        "pct_small": float(np.mean(areas < small_thr)),
        "pct_large": float(np.mean(areas > large_thr)),
    }


def _print_params(params):
    keys = [
        "detector_backend",
        "model_name",
        "prob_thresh",
        "nms_thresh",
        "scale",
        "min_area_px",
        "max_area_px",
        "purple_filter_enabled",
        "strict_purple_filter",
        "require_center_purple",
        "purple_h_min",
        "purple_h_max",
        "purple_s_min",
        "purple_v_max",
        "min_purple_ratio",
    ]
    print("Detection params:")
    for key in keys:
        if key in params:
            print(f"  {key}: {params[key]}")


def _print_diagnostics(diag, small_area_px, large_area_px):
    model_before = diag.get("model_candidates_before_nms")
    before_text = "n/a" if model_before is None else str(int(model_before))

    print("Diagnostics:")
    print(f"  backend: {diag.get('backend', '')}")
    print(f"  model candidates before NMS: {before_text}")
    print(f"  after NMS: {int(diag.get('after_nms_count', 0))}")
    print(f"  removed by min_area: {int(diag.get('removed_by_min_area', 0))}")
    print(f"  removed by max_area/additional filters: {int(diag.get('removed_by_max_area', 0))}")

    if bool(diag.get("purple_filter_enabled", False)):
        print(f"  purple checked: {int(diag.get('purple_checked', 0))}")
        print(f"  removed by purple total: {int(diag.get('removed_by_purple_total', 0))}")
        print(f"    removed by purple center: {int(diag.get('removed_by_purple_center', 0))}")
        print(f"    removed by purple ratio: {int(diag.get('removed_by_purple_ratio', 0))}")
        print(f"    removed by purple other: {int(diag.get('removed_by_purple_other', 0))}")
        if bool(diag.get("strict_purple_filter", True)):
            print("  purple mode: strict")
        else:
            print("  purple mode: soft")
            print(f"  weak_color marked: {int(diag.get('weak_color_marked', 0))}")
    else:
        print("  purple filter: disabled")

    print(f"  final objects: {int(diag.get('final_count', 0))}")
    area_stats = diag.get("final_area_stats", {}) or {}
    print(
        "  area stats px: "
        f"min={float(area_stats.get('min', 0.0)):.2f}, "
        f"median={float(area_stats.get('median', 0.0)):.2f}, "
        f"mean={float(area_stats.get('mean', 0.0)):.2f}, "
        f"max={float(area_stats.get('max', 0.0)):.2f}"
    )
    small_pct = 100.0 * float(area_stats.get("pct_small", 0.0))
    large_pct = 100.0 * float(area_stats.get("pct_large", 0.0))
    print(
        f"  small<{float(small_area_px):.1f}px: {small_pct:.2f}% | "
        f"large>{float(large_area_px):.1f}px: {large_pct:.2f}%"
    )


def _print_tiled_report(report, nuclei, small_area_px, large_area_px):
    print("Sliced inference report:")
    print(f"  average_cell_diameter_px: {float(report.get('average_cell_diameter_px', 0.0)):.3f}")
    print(f"  scale: {float(report.get('scale', 1.0)):.3f}")
    print(f"  tile_size_px (infer): {int(report.get('tile_size_px', 0))}")
    print(f"  stride_px (infer): {int(report.get('stride_px', 0))}")
    print(f"  tile_size_px (orig): {int(report.get('tile_size_orig_px', 0))}")
    print(f"  stride_px (orig): {int(report.get('stride_orig_px', 0))}")
    print(f"  tiles total/used: {int(report.get('tiles_total', 0))}/{int(report.get('tiles_used', 0))}")
    print(
        f"  candidates before ROI filter: {int(report.get('candidates_before_roi_filter', 0))} | "
        f"in ROI: {int(report.get('candidates_in_roi', 0))} | final: {int(report.get('final_count', 0))}"
    )

    for roi_stat in report.get("roi_stats", []):
        print(
            f"  ROI {roi_stat.get('roi_id')}: kind={roi_stat.get('roi_kind')} "
            f"area_px={float(roi_stat.get('roi_area_px', 0.0)):.2f} "
            f"area_cells={float(roi_stat.get('roi_area_in_cells', 0.0)):.3f} "
            f"tiles={int(roi_stat.get('tiles_total', 0))}"
        )

    area_stats = _area_stats(nuclei, small_area_px, large_area_px)
    print(
        "  area stats px: "
        f"min={area_stats['min']:.2f}, median={area_stats['median']:.2f}, "
        f"mean={area_stats['mean']:.2f}, max={area_stats['max']:.2f}"
    )
    print(
        f"  small<{float(small_area_px):.1f}px: {area_stats['pct_small'] * 100.0:.2f}% | "
        f"large>{float(large_area_px):.1f}px: {area_stats['pct_large'] * 100.0:.2f}%"
    )


def _find_images(images_dir):
    root = Path(images_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Image directory does not exist: {images_dir}")

    files = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() in backend.SUPPORTED_IMAGE_FORMATS:
            files.append(p)
    return files


def _draw_overlay(image_bgr, nuclei):
    out = image_bgr.copy()
    for nucleus in nuclei:
        contour = nucleus.get("contour", [])
        if len(contour) >= 3:
            arr = np.asarray(contour, dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(out, [arr], -1, (0, 255, 0), 1)
        center = nucleus.get("center")
        if center is not None:
            cx = int(round(float(center[0])))
            cy = int(round(float(center[1])))
            cv2.circle(out, (cx, cy), 1, (0, 0, 255), -1)
    return out


def _save_json_result(output_path, image_path, nuclei, params, diagnostics=None, tiled_report=None):
    payload = {
        "image": str(image_path),
        "count": int(len(nuclei)),
        "nuclei": nuclei,
        "detection_params": params,
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    if tiled_report is not None:
        payload["tiled_report"] = tiled_report

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_infer(args):
    config_data = {}
    if args.config:
        config_data = _load_config(args.config)

    params = backend.build_detection_params_for_preset(args.preset)
    params.update(config_data)
    params.update(_collect_overrides(args))
    backend.set_detection_params(params)
    params = backend.get_detection_params()

    if args.print_params:
        _print_params(params)

    run_opts = _collect_tiling_options(args, config_data)
    diag_enabled = bool(args.diagnostics or args.debug_filters)

    nuclei = []
    diagnostics = None
    tiled_report = None

    if run_opts["sliced_inference"]:
        nuclei, tiled_report = backend.detect_nuclei_tiled(
            image_path=args.image,
            average_cell_diameter_px=run_opts.get("average_cell_diameter_px"),
            average_cell_bbox=run_opts.get("average_cell_bbox"),
            enhancement_params=None,
            tile_factor=float(run_opts.get("tile_factor", 64.0)),
            overlap_ratio=float(run_opts.get("overlap_ratio", 0.25)),
            tile_min_px=int(run_opts.get("tile_min_px", 512)),
            tile_max_px=int(run_opts.get("tile_max_px", 2048)),
            min_roi_cover_ratio=float(run_opts.get("min_roi_cover_ratio", 0.05)),
            roi_rect=run_opts.get("roi_rect", []),
            roi_poly=run_opts.get("roi_poly", []),
            merge_iou_thresh=float(run_opts.get("merge_iou_thresh", 0.30)),
            border_trim_enabled=bool(run_opts.get("border_trim_enabled", True)),
            border_trim_factor=float(run_opts.get("border_trim_factor", 1.0)),
        )
        print(f"Detected objects: {len(nuclei)}")
        if diag_enabled:
            _print_tiled_report(tiled_report, nuclei, args.small_area_px, args.large_area_px)
    else:
        if diag_enabled:
            nuclei, diagnostics = backend.detect_nuclei_with_diagnostics(
                args.image,
                enhancement_params=None,
                area_small_px=float(args.small_area_px),
                area_large_px=float(args.large_area_px),
            )
            print(f"Detected objects: {len(nuclei)}")
            _print_diagnostics(diagnostics, args.small_area_px, args.large_area_px)
        else:
            nuclei = backend.detect_nuclei(args.image)
            print(f"Detected objects: {len(nuclei)}")

    if args.output_json:
        _save_json_result(
            output_path=args.output_json,
            image_path=args.image,
            nuclei=nuclei,
            params=params,
            diagnostics=diagnostics,
            tiled_report=tiled_report,
        )
        print(f"Saved JSON results: {args.output_json}")

    if args.output_overlay:
        image = backend.load_image(args.image)
        overlay = _draw_overlay(image, nuclei)
        ok = cv2.imwrite(str(args.output_overlay), overlay)
        if not ok:
            raise RuntimeError(f"Failed to save overlay: {args.output_overlay}")
        print(f"Saved overlay: {args.output_overlay}")


def _run_tune(args):
    image_paths = _find_images(args.images_dir)
    if not image_paths:
        raise ValueError("No supported images found in directory")

    num_images = max(1, int(args.num_images))
    selected = image_paths[:num_images]

    base_params = backend.build_detection_params_for_preset(args.preset)
    base_params["detector_backend"] = args.detector_backend
    if args.model_name:
        base_params["model_name"] = args.model_name

    if args.strict_purple_filter:
        base_params["strict_purple_filter"] = True
    elif args.soft_purple_filter:
        base_params["strict_purple_filter"] = False

    prob_thresh_grid = _parse_float_grid(args.prob_thresh_grid)
    nms_thresh_grid = _parse_float_grid(args.nms_thresh_grid)
    min_area_grid = _parse_int_grid(args.min_area_grid)
    scale_grid = _parse_float_grid(args.scale_grid)
    purple_enabled_grid = _parse_bool_grid(args.purple_enabled_grid)
    require_center_grid = _parse_bool_grid(args.require_center_grid)
    min_purple_ratio_grid = _parse_float_grid(args.min_purple_ratio_grid)
    purple_s_min_grid = _parse_int_grid(args.purple_s_min_grid)
    purple_v_max_grid = _parse_int_grid(args.purple_v_max_grid)

    combos = list(
        itertools.product(
            prob_thresh_grid,
            nms_thresh_grid,
            min_area_grid,
            scale_grid,
            purple_enabled_grid,
            require_center_grid,
            min_purple_ratio_grid,
            purple_s_min_grid,
            purple_v_max_grid,
        )
    )

    if args.max_combinations > 0:
        combos = combos[: int(args.max_combinations)]

    results = []
    print(f"Tuning: {len(combos)} combinations, {len(selected)} images")

    for idx, combo in enumerate(combos, start=1):
        (
            prob_thresh,
            nms_thresh,
            min_area_px,
            scale,
            purple_filter_enabled,
            require_center_purple,
            min_purple_ratio,
            purple_s_min,
            purple_v_max,
        ) = combo

        params = dict(base_params)
        params.update(
            {
                "prob_thresh": float(prob_thresh),
                "nms_thresh": float(nms_thresh),
                "min_area_px": int(min_area_px),
                "scale": float(scale),
                "purple_filter_enabled": bool(purple_filter_enabled),
                "require_center_purple": bool(require_center_purple),
                "min_purple_ratio": float(min_purple_ratio),
                "purple_s_min": int(purple_s_min),
                "purple_v_max": int(purple_v_max),
            }
        )

        backend.set_detection_params(params)

        per_image_counts = []
        all_areas = []
        tiny_count = 0
        total_count = 0
        purple_metric_values = []

        for image_path in selected:
            nuclei, diag = backend.detect_nuclei_with_diagnostics(
                str(image_path),
                enhancement_params=None,
                area_small_px=float(args.small_area_px),
                area_large_px=float(args.large_area_px),
            )

            count = len(nuclei)
            per_image_counts.append(count)

            tiny_thr = max(0.0, float(params["min_area_px"]) * 1.5)
            for nucleus in nuclei:
                area = float(nucleus.get("area_px", 0.0))
                all_areas.append(area)
                total_count += 1
                if area < tiny_thr:
                    tiny_count += 1

            if bool(params.get("purple_filter_enabled", False)):
                denom = max(1, int(diag.get("after_area_filters_count", 0)))
                if bool(params.get("strict_purple_filter", True)):
                    purple_value = float(diag.get("removed_by_purple_total", 0)) / float(denom)
                else:
                    purple_value = float(diag.get("weak_color_marked", 0)) / float(denom)
                purple_metric_values.append(purple_value)

        avg_objects = float(np.mean(per_image_counts)) if per_image_counts else 0.0
        std_objects = float(np.std(per_image_counts)) if per_image_counts else 0.0
        tiny_share = float(tiny_count) / float(max(1, total_count))
        median_area = float(np.median(np.asarray(all_areas, dtype=np.float32))) if all_areas else 0.0
        purple_metric = float(np.mean(purple_metric_values)) if purple_metric_values else 0.0

        penalty = 0.0
        if tiny_share > float(args.max_tiny_share):
            penalty = (tiny_share - float(args.max_tiny_share)) * (avg_objects + 1.0) * 3.0
        score = avg_objects - penalty

        metrics = {
            "avg_objects_per_image": avg_objects,
            "std_objects_per_image": std_objects,
            "small_objects_share": tiny_share,
            "median_area": median_area,
            "purple_rejected_share_or_weak_color_share": purple_metric,
        }

        results.append(
            {
                "score": score,
                "penalty": penalty,
                "params": params,
                "metrics": metrics,
            }
        )

        if idx % 10 == 0 or idx == len(combos):
            print(f"  processed {idx}/{len(combos)}")

    if not results:
        raise RuntimeError("No tuning results were produced")

    results.sort(key=lambda item: item["score"], reverse=True)
    top_n = min(10, len(results))

    print("Top configs:")
    for rank in range(top_n):
        item = results[rank]
        params = item["params"]
        metrics = item["metrics"]
        print(
            f"{rank + 1:2d}. score={item['score']:.3f} "
            f"avg={metrics['avg_objects_per_image']:.2f} std={metrics['std_objects_per_image']:.2f} "
            f"tiny={metrics['small_objects_share']:.3f} med_area={metrics['median_area']:.2f} "
            f"purple={metrics['purple_rejected_share_or_weak_color_share']:.3f} "
            f"prob={params['prob_thresh']} nms={params['nms_thresh']} min_area={params['min_area_px']} "
            f"scale={params['scale']} purple_enabled={int(bool(params['purple_filter_enabled']))} "
            f"center={int(bool(params['require_center_purple']))} sat={params['purple_s_min']} "
            f"vmax={params['purple_v_max']} ratio={params['min_purple_ratio']}"
        )

    best = results[0]
    output_path = Path(args.output_config)
    payload = dict(best["params"])
    payload["_meta"] = {
        "score": best["score"],
        "penalty": best["penalty"],
        "metrics": best["metrics"],
        "preset": args.preset,
        "num_images": len(selected),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved best config: {output_path}")
    first_image = str(selected[0])
    print("Run inference with tuned config:")
    print(
        "  "
        f"python nuclei_cli.py infer --image \"{first_image}\" "
        f"--config \"{output_path}\" --diagnostics"
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Local CLI for StarDist/Cellpose nuclei inference, diagnostics and tuning"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run inference on a single image")
    infer.add_argument("--image", required=True, help="Path to input image")
    infer.add_argument(
        "--preset",
        default="legacy",
        choices=backend.get_runtime_detection_presets(),
        help="Runtime preset for postprocess and thresholds",
    )
    infer.add_argument("--config", default="", help="Path to JSON config")
    infer.add_argument("--detector-backend", default="stardist", choices=backend.get_detector_backends())
    infer.add_argument("--model-name", default="", help="Pretrained StarDist model name")

    infer.add_argument("--prob-thresh", type=float, default=None)
    infer.add_argument("--nms-thresh", type=float, default=None)
    infer.add_argument("--scale", type=float, default=None)
    infer.add_argument("--min-area-px", type=int, default=None)
    infer.add_argument("--max-area-px", type=int, default=None)
    infer.add_argument("--min-purple-ratio", type=float, default=None)
    infer.add_argument("--purple-h-min", type=int, default=None)
    infer.add_argument("--purple-h-max", type=int, default=None)
    infer.add_argument("--purple-s-min", type=int, default=None)
    infer.add_argument("--purple-v-max", type=int, default=None)

    infer.add_argument("--disable-purple-filter", action="store_true")
    infer.add_argument("--enable-purple-filter", action="store_true")
    infer.add_argument("--require-center-purple", action="store_true")
    infer.add_argument("--no-require-center-purple", action="store_true")
    infer.add_argument("--strict-purple-filter", action="store_true")
    infer.add_argument("--soft-purple-filter", action="store_true")

    infer.add_argument("--diagnostics", action="store_true")
    infer.add_argument("--debug-filters", action="store_true")
    infer.add_argument("--small-area-px", type=float, default=15.0)
    infer.add_argument("--large-area-px", type=float, default=2000.0)
    infer.add_argument("--print-params", action="store_true")

    infer.add_argument("--sliced-inference", action="store_true")
    infer.add_argument("--average-cell-diameter-px", type=float, default=None)
    infer.add_argument("--average-cell-bbox", default="", help="x,y,w,h")
    infer.add_argument("--tile-factor", type=float, default=None)
    infer.add_argument("--overlap-ratio", type=float, default=None)
    infer.add_argument("--tile-min-px", type=int, default=None)
    infer.add_argument("--tile-max-px", type=int, default=None)
    infer.add_argument("--min-roi-cover-ratio", type=float, default=None)
    infer.add_argument("--roi-rect", action="append", default=[], help="x,y,w,h. Can be repeated")
    infer.add_argument(
        "--roi-poly",
        action="append",
        default=[],
        help="x1,y1,x2,y2,... polygon in pixels. Can be repeated",
    )
    infer.add_argument("--roi-file", default="", help="Path to ROI JSON/GeoJSON")
    infer.add_argument("--merge-iou-thresh", type=float, default=None)
    infer.add_argument("--border-trim-enabled", action="store_true")
    infer.add_argument("--border-trim-disabled", action="store_true")
    infer.add_argument("--border-trim-factor", type=float, default=None)

    infer.add_argument("--output-json", default="", help="Save final detections to JSON")
    infer.add_argument("--output-overlay", default="", help="Save optional overlay image")

    tune = subparsers.add_parser("tune", help="Grid-search params on local image folder")
    tune.add_argument("--images-dir", required=True, help="Folder with images")
    tune.add_argument("--num-images", type=int, default=3, help="Use first N images for tuning")
    tune.add_argument(
        "--preset",
        default="balanced",
        choices=backend.get_runtime_detection_presets(),
        help="Base runtime preset",
    )
    tune.add_argument("--detector-backend", default="stardist", choices=backend.get_detector_backends())
    tune.add_argument("--model-name", default="", help="Pretrained StarDist model name")

    tune.add_argument("--strict-purple-filter", action="store_true")
    tune.add_argument("--soft-purple-filter", action="store_true")

    tune.add_argument("--prob-thresh-grid", default="0.05,0.1,0.15")
    tune.add_argument("--nms-thresh-grid", default="0.3,0.4,0.55")
    tune.add_argument("--min-area-grid", default="0,10,18")
    tune.add_argument("--scale-grid", default="0.75,1.0,1.25")
    tune.add_argument("--purple-enabled-grid", default="0,1")
    tune.add_argument("--require-center-grid", default="0,1")
    tune.add_argument("--min-purple-ratio-grid", default="0.05,0.1,0.2")
    tune.add_argument("--purple-s-min-grid", default="15,30,45")
    tune.add_argument("--purple-v-max-grid", default="230,255")

    tune.add_argument("--small-area-px", type=float, default=15.0)
    tune.add_argument("--large-area-px", type=float, default=2000.0)
    tune.add_argument(
        "--max-tiny-share",
        type=float,
        default=0.35,
        help="Penalty threshold for small object share",
    )
    tune.add_argument("--max-combinations", type=int, default=0, help="0 = full grid")
    tune.add_argument("--output-config", default="tuned_config.json")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "infer":
            _run_infer(args)
            return
        if args.command == "tune":
            _run_tune(args)
            return
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
