#!/usr/bin/env python3
import argparse

import backend


def main():
    parser = argparse.ArgumentParser(description="Minimal smoke test for nuclei inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--preset",
        default="legacy",
        choices=backend.get_runtime_detection_presets(),
        help="Runtime preset",
    )
    parser.add_argument("--detector-backend", default="stardist", choices=backend.get_detector_backends())
    parser.add_argument("--model-name", default="", help="Pretrained StarDist model name")
    args = parser.parse_args()

    params = backend.build_detection_params_for_preset(args.preset)
    params["detector_backend"] = args.detector_backend
    if args.model_name:
        params["model_name"] = args.model_name

    backend.set_detection_params(params)
    try:
        nuclei = backend.detect_nuclei(args.image)
    except Exception as exc:
        print(f"SMOKE_FAIL preset={args.preset} error={exc}")
        raise SystemExit(1) from exc
    print(f"SMOKE_OK preset={args.preset} nuclei_count={len(nuclei)}")


if __name__ == "__main__":
    main()
