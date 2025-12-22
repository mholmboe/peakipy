"""
Batch fitting script for the Profile Fitting Application.

Usage:
    python batch_fit.py "data/*.txt" --baseline asls --lam 1e5 --p 0.01 \
        --profile gaussian --components 2 --centers 10,20 --sigmas 1,1 \
        --amplitudes 1,0.8 --normalize --interp_step 0.1

Notes:
- Preprocessing: optional cropping (--x_min/--x_max) and interpolation to regular spacing (--interp_step).
- Baseline options:
    asls:      --lam, --p
    polynomial:--degree
    linear:    --slope, --intercept (optional; auto-fit if omitted)
    rolling_ball: --radius
    shirley:   --tol, --max_iter, --start_offset, --end_offset
    manual:    --manual_points x1:y1,x2:y2,...  --manual_interp linear|cubic
- Components: provide per-component lists for centers, widths/sigmas/gammas, amplitudes.
- Outputs: for each file, writes <base>_results.txt and <base>_data.txt alongside the input file.
"""

import argparse
import glob
import os
import sys
import numpy as np

from core.data_import import load_data_file
from core.fitting import ProfileFitter
from core.fitting.model_builder import build_composite_model
from core.data_preprocessing import crop_roi, interpolate_data


def parse_args():
    p = argparse.ArgumentParser(description="Batch profile fitting")
    p.add_argument("pattern", help="Glob pattern for data files, e.g. 'data/*.txt'")

    # Baseline
    p.add_argument("--baseline", default=None, choices=["asls", "polynomial", "linear", "rolling_ball", "shirley", "manual"], help="Baseline method")
    p.add_argument("--lam", type=float, default=1e5, help="ASLS lambda")
    p.add_argument("--p", type=float, default=0.01, help="ASLS p")
    p.add_argument("--degree", type=int, default=2, help="Polynomial degree")
    p.add_argument("--slope", type=float, default=None, help="Linear slope (optional)")
    p.add_argument("--intercept", type=float, default=None, help="Linear intercept (optional)")
    p.add_argument("--radius", type=float, default=50.0, help="Rolling ball radius")
    p.add_argument("--tol", type=float, default=1e-5, help="Shirley tolerance")
    p.add_argument("--max_iter", type=int, default=50, help="Shirley max iterations")
    p.add_argument("--start_offset", type=float, default=0.0, help="Shirley start offset")
    p.add_argument("--end_offset", type=float, default=0.0, help="Shirley end offset")
    p.add_argument("--manual_points", type=str, default=None, help="Manual baseline control points x:y pairs, e.g., 0:0.1,5:0.2,10:0.15")
    p.add_argument("--manual_interp", type=str, default="linear", choices=["linear", "cubic"], help="Manual baseline interpolation")

    # Baseline range
    p.add_argument("--calc_min", type=float, default=None, help="Baseline calculation range min (flattened outside)")
    p.add_argument("--calc_max", type=float, default=None, help="Baseline calculation range max (flattened outside)")
    p.add_argument("--fit_min", type=float, default=None, help="Optional fit/evaluation range min (if you want to crop before fitting)")
    p.add_argument("--fit_max", type=float, default=None, help="Optional fit/evaluation range max (if you want to crop before fitting)")
    p.add_argument("--x_min", type=float, default=None, help="Crop data min X")
    p.add_argument("--x_max", type=float, default=None, help="Crop data max X")
    p.add_argument("--interp_step", type=float, default=None, help="Interpolate data to regular step (e.g., 0.1). If omitted, keep original spacing.")

    # Components
    p.add_argument("--profile", default="gaussian", choices=["gaussian", "lorentzian", "voigt"], help="Profile type for all components")
    p.add_argument("--components", type=int, default=1, help="Number of components")
    p.add_argument("--centers", type=str, default=None, help="Comma-separated centers")
    p.add_argument("--widths", type=str, default=None, help="Comma-separated widths (sigma for gaussian, gamma for lorentzian/voigt)")
    p.add_argument("--sigmas", type=str, default=None, help="Comma-separated sigmas (for gaussian/voigt)")
    p.add_argument("--gammas", type=str, default=None, help="Comma-separated gammas (for lorentzian/voigt)")
    p.add_argument("--amplitudes", type=str, default=None, help="Comma-separated amplitudes")

    # Options
    p.add_argument("--normalize", action="store_true", help="Normalize raw data before fitting")
    p.add_argument("--optimize_baseline", action="store_true", help="Optimize baseline simultaneously with peaks")
    p.add_argument("--max_nfev", type=int, default=None, help="Max function evals for optimizer")

    return p.parse_args()


def parse_list(arg, n):
    if arg is None:
        return [None] * n
    parts = [float(x) for x in arg.split(",")]
    if len(parts) != n:
        raise ValueError(f"Expected {n} values, got {len(parts)}")
    return parts


def parse_points(arg):
    pts = []
    if not arg:
        return pts
    for pair in arg.split(","):
        x_str, y_str = pair.split(":")
        pts.append((float(x_str), float(y_str)))
    return pts


def export_results(base_path, fitter, result):
    results_file = f"{base_path}_results.txt"
    data_file = f"{base_path}_data.txt"

    with open(results_file, "w") as f:
        f.write(fitter.get_fit_report())
        f.write("\n\nStatistics:\n")
        for k, v in fitter.get_statistics().items():
            f.write(f"{k}: {v}\n")

    x = fitter.x
    y_exp = fitter.y_corrected
    y_fit = result.best_fit
    res = result.residual
    comp_evals = result.eval_components(x=x)
    comp_keys = sorted([k for k in comp_evals.keys() if k.startswith("c")])

    header = "X\tY_Exp\tY_Fit\tResidual" + "".join([f"\tComp_{i+1}" for i in range(len(comp_keys))])
    with open(data_file, "w") as f:
        f.write(header + "\n")
        for i in range(len(x)):
            line = f"{x[i]:.6e}\t{y_exp[i]:.6e}\t{y_fit[i]:.6e}\t{res[i]:.6e}"
            for k in comp_keys:
                line += f"\t{comp_evals[k][i]:.6e}"
            f.write(line + "\n")


def build_fitter(x, y, args):
    fitter = ProfileFitter(x, y)

    # Baseline setup
    if args.baseline:
        bl_params = {}
        if args.baseline == "asls":
            bl_params.update({"lam": args.lam, "p": args.p})
        elif args.baseline == "polynomial":
            bl_params.update({"degree": args.degree})
        elif args.baseline == "linear":
            if args.slope is not None and args.intercept is not None:
                bl_params.update({"slope": args.slope, "intercept": args.intercept})
        elif args.baseline == "rolling_ball":
            bl_params.update({"radius": args.radius})
        elif args.baseline == "shirley":
            bl_params.update({"tol": args.tol, "max_iter": args.max_iter,
                              "start_offset": args.start_offset, "end_offset": args.end_offset})
        elif args.baseline == "manual":
            pts = parse_points(args.manual_points)
            bl_params.update({"points": pts, "interp": args.manual_interp})

        fitter.set_baseline(args.baseline, **bl_params)
        fitter.baseline_range = (args.calc_min, args.calc_max)
        fitter.optimize_baseline = args.optimize_baseline

    # Components
    n = args.components
    centers = parse_list(args.centers, n)
    amplitudes = parse_list(args.amplitudes, n)
    widths = parse_list(args.widths, n)
    sigmas = parse_list(args.sigmas, n)
    gammas = parse_list(args.gammas, n)

    for i in range(n):
        prefix = f"c{i+1}_"
        center = centers[i] if centers[i] is not None else 0
        amp = amplitudes[i] if amplitudes[i] is not None else 1.0
        if args.profile == "voigt":
            sigma = sigmas[i] if sigmas[i] is not None else (widths[i] if widths[i] is not None else 1.0)
            gamma = gammas[i] if gammas[i] is not None else (widths[i] if widths[i] is not None else 1.0)
            fitter.add_component(args.profile, prefix=prefix, amplitude=amp, center=center, sigma=sigma, gamma=gamma)
        elif args.profile == "gaussian":
            width = sigmas[i] if sigmas[i] is not None else (widths[i] if widths[i] is not None else 1.0)
            fitter.add_component(args.profile, prefix=prefix, amplitude=amp, center=center, sigma=width)
        else:
            width = gammas[i] if gammas[i] is not None else (widths[i] if widths[i] is not None else 1.0)
            fitter.add_component(args.profile, prefix=prefix, amplitude=amp, center=center, gamma=width)

    return fitter


def main():
    args = parse_args()
    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matched pattern: {args.pattern}")
        sys.exit(1)

    for fname in files:
        try:
            x, y = load_data_file(fname)
            # Crop and interpolate if requested
            xmin = args.x_min if args.x_min is not None else args.fit_min
            xmax = args.x_max if args.x_max is not None else args.fit_max
            if xmin is not None or xmax is not None:
                x, y = crop_roi(x, y, x_min=xmin, x_max=xmax)
            if args.interp_step is not None:
                x, y = interpolate_data(x, y, x_min=xmin, x_max=xmax, step=args.interp_step)
            fitter = build_fitter(x, y, args)
            if args.normalize:
                fitter.normalize_raw_data()
            # pre-apply baseline for plotting consistency; respect calc range
            if fitter.baseline_method:
                fitter.apply_baseline_correction()
            if args.normalize:
                fitter.normalize_intensity()

            result = fitter.fit(max_nfev=args.max_nfev, skip_baseline_correction=False)

            base, _ = os.path.splitext(fname)
            export_results(base, fitter, result)
            print(f"Processed {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
