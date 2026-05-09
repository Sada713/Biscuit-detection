"""
Biscuit Classifier
==================

CORE INSIGHT:
  A broken circle fragment's minimum enclosing circle radius equals the
  radius of the intact circle it came from. A broken rectangle's enclosing
  circle radius is just half its diagonal — much smaller than any biscuit circle.

  So in a mixed image: r_ratio = piece_r / bench_circle_r
    broken circle  -> r_ratio close to 1.0   (0.80–1.50)
    broken rect    -> r_ratio much less       (0.40–0.65)

FULL DECISION TREE:

  BENCHMARK PASS (per image):
    INTACT CIRCLE : arc_fraction >= 0.75
    INTACT RECT   : arc_fraction <  0.75  AND  solidity >= 0.96
    has_circles / has_rects flags set accordingly

  SHAPE (per blob):
    af >= 0.75                       -> Circle  (intact)
    not has_rects                    -> Circle  (only circle type in image)
    not has_circles                  -> Rectangle (only rect type in image)
    both present, r_ratio >= 0.80   -> Circle  (enclosing radius matches circle benchmark)
    both present, r_ratio <  0.80   -> Rectangle

  STATE:
    Circle:    INTACT if af >= 0.75 AND size_ratio >= 0.75
               else BROKEN
    Rectangle: INTACT if perp_ratio >= 0.55 AND size_ratio >= 0.65
               else BROKEN
"""

import os, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── hresholdsT ──────────────────────────────────────────────────────────────

ARC_INTACT_MIN   = 0.75   # arc_fraction for intact circle detection
ARC_TOLERANCE    = 0.15   # within 15% of radius -> on arc
SOL_INTACT_RECT  = 0.96   # solidity floor for intact rect detection

R_RATIO_CIRCLE   = 0.67   # piece_r / bench_r >= this -> circle family (mixed images)
                            # Gap in data: circles >= 0.715, rects <= 0.623

CIRCLE_SIZE_MIN  = 0.75   # size_ratio (r/bench_r)^2 for intact circle
RECT_PERP_MIN    = 0.55   # perp_ratio for intact rectangle
RECT_SIZE_MIN    = 0.65   # size_ratio (area/bench_area) for intact rectangle

LINE_THRESHOLD   = 20
LINE_MIN_LEN     = 40
LINE_GAP         = 10
ANGLE_BIN        = 15
PERP_TOL         = 15
MIN_BLOB_AREA    = 5000


# ── Masking ─────────────────────────────────────────────────────────────────

def get_paper_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.ones(gray.shape, np.uint8) * 255
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.erode(mask, np.ones((20, 20), np.uint8))


def get_biscuit_mask(img, pmask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m   = cv2.inRange(hsv, np.array([5, 20, 30]), np.array([45, 255, 255]))
    m   = cv2.bitwise_and(m, pmask)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


# ── Geometry ─────────────────────────────────────────────────────────────────

def arc_fraction(cnt):
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    if r == 0:
        return 0.0, 0.0
    pts   = cnt[:, 0, :].astype(float)
    dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return np.sum(np.abs(dists - r) < r * ARC_TOLERANCE) / len(pts), r


def solidity(cnt):
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return cv2.contourArea(cnt) / hull_area if hull_area > 0 else 1.0


def perp_line_ratio(cnt, img_shape):
    blob = np.zeros(img_shape[:2], np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, 2)
    lines = cv2.HoughLinesP(blob, 1, np.pi / 180,
                             threshold=LINE_THRESHOLD,
                             minLineLength=LINE_MIN_LEN,
                             maxLineGap=LINE_GAP)
    if lines is None:
        return 0.0, 0.0
    buckets = {}
    for l in lines:
        x1, y1, x2, y2 = l[0]
        ang    = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.hypot(x2 - x1, y2 - y1)
        b      = round(ang / ANGLE_BIN) * ANGLE_BIN % 180
        buckets[b] = buckets.get(b, 0.0) + length
    sorted_b         = sorted(buckets.items(), key=lambda x: -x[1])
    dom_ang, dom_len = sorted_b[0]
    perp_len         = 0.0
    for ang2, len2 in sorted_b[1:]:
        diff = min(abs(dom_ang - ang2) % 180, 180 - abs(dom_ang - ang2) % 180)
        if (90 - PERP_TOL) <= diff <= (90 + PERP_TOL):
            perp_len = len2
            break
    return dom_len, perp_len


# ── Benchmark pass ───────────────────────────────────────────────────────────

def compute_benchmarks(blobs):
    circle_radii, circle_areas, rect_areas, all_radii = [], [], [], []

    for area, cnt in blobs:
        af, r = arc_fraction(cnt)
        sol   = solidity(cnt)
        all_radii.append(r)

        if af >= ARC_INTACT_MIN:
            circle_radii.append(r)
            circle_areas.append(np.pi * r ** 2)
        elif sol >= SOL_INTACT_RECT:
            # af < ARC_INTACT_MIN is implicit — circles already caught above
            rect_areas.append(area)

    has_circles = len(circle_radii) > 0
    has_rects   = len(rect_areas)   > 0

    bench_r = (float(np.median(circle_radii)) if has_circles
               else float(max(all_radii, default=1)))

    if has_rects:
        bench_area = float(np.median(sorted(rect_areas, reverse=True)[:3]))
    elif has_circles:
        bench_area = float(np.median(circle_areas))
    else:
        bench_area = float(np.median(sorted([a for a, _ in blobs],
                                            reverse=True)[:3])) if blobs else 1.0

    return bench_r, bench_area, has_circles, has_rects


# ── Classification ───────────────────────────────────────────────────────────

def classify(cnt, bench_r, bench_area, has_circles, has_rects, img_shape):
    af, r  = arc_fraction(cnt)
    sol    = solidity(cnt)
    area   = cv2.contourArea(cnt)
    r_ratio = r / bench_r if bench_r > 0 else 0.0

    # ── Shape ──────────────────────────────────────────────────────────────
    if af >= ARC_INTACT_MIN:
        shape = "Circle"

    elif not has_rects:
        # Only circle biscuits present — all broken pieces must be circles
        shape = "Circle"

    elif not has_circles:
        # Only rectangle biscuits present — all broken pieces must be rects
        shape = "Rectangle"

    else:
        # Both types present: use enclosing-circle radius ratio.
        # A broken circle fragment came from an intact circle, so its
        # enclosing radius equals the intact circle's radius (r_ratio ~ 1.0).
        # A broken rect's enclosing radius is just half its diagonal (r_ratio << 1).
        shape = "Circle" if r_ratio >= R_RATIO_CIRCLE else "Rectangle"

    # ── State ──────────────────────────────────────────────────────────────
    if shape == "Circle":
        size_ratio = (r / bench_r) ** 2 if bench_r > 0 else 1.0
        state      = "INTACT" if (af >= ARC_INTACT_MIN and size_ratio >= CIRCLE_SIZE_MIN) else "BROKEN"
        perp_ratio = None

    else:
        size_ratio = area / bench_area if bench_area > 0 else 1.0
        dom, perp  = perp_line_ratio(cnt, img_shape)
        perp_ratio = perp / dom if dom > 0 else 0.0
        state      = "INTACT" if (perp_ratio >= RECT_PERP_MIN and size_ratio >= RECT_SIZE_MIN) else "BROKEN"

    metrics = dict(af=af, sol=sol, r_ratio=r_ratio, size=size_ratio, perp_ratio=perp_ratio)
    return shape, state, metrics


# ── Visualisation ─────────────────────────────────────────────────────────────

COLORS = {
    ("INTACT", "Circle"):    "#00e676",
    ("INTACT", "Rectangle"): "#69f0ae",
    ("BROKEN", "Circle"):    "#ff1744",
    ("BROKEN", "Rectangle"): "#ff6d00",
}


def draw_ghost(ax, cnt, shape, bench_r, bench_area):
    (cx, cy), _      = cv2.minEnclosingCircle(cnt)
    _, (w, h), angle = cv2.minAreaRect(cnt)
    if shape == "Circle" and bench_r > 0:
        ax.add_patch(plt.Circle((cx, cy), bench_r,
                                color='cyan', fill=False,
                                linestyle='--', linewidth=1.2, alpha=0.5))
    elif shape == "Rectangle" and bench_area > 0 and w > 0 and h > 0:
        scale = np.sqrt(bench_area / (w * h))
        ghost = ((cx, cy), (w * scale, h * scale), angle)
        box   = cv2.boxPoints(ghost).astype(int)
        pts   = np.vstack([box, box[0]])
        ax.plot(pts[:, 0], pts[:, 1],
                color='cyan', linestyle='--', linewidth=1.2, alpha=0.5)


def annotate(ax, cnt, shape, state, metrics):
    color  = COLORS[(state, shape)]
    xs, ys = cnt[:, 0, 0], cnt[:, 0, 1]
    ax.plot(xs, ys, color=color, linewidth=2.5)
    if shape == "Circle":
        detail = f"arc={metrics['af']:.2f}  r_ratio={metrics['r_ratio']:.2f}"
    else:
        pr     = metrics['perp_ratio']
        detail = f"perp={pr:.2f}  size={metrics['size']:.2f}" if pr is not None else f"size={metrics['size']:.2f}"
    ax.text(xs.mean(), ys.mean(),
            f"{state} {shape}\n{detail}",
            color='white', fontsize=7.5, fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor=color, alpha=0.75,
                      edgecolor='none', boxstyle='round,pad=0.3'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    img_dir = r"C:\Users\ASUS\Desktop\Semester 7\Image processing\Assignment_2\Images"

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(os.path.join(img_dir, filename))
        if img is None:
            continue

        t0      = time.time()
        pmask   = get_paper_mask(img)
        bmask   = get_biscuit_mask(img, pmask)
        cnts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        blobs   = sorted(
            [(cv2.contourArea(c), c) for c in cnts if cv2.contourArea(c) >= MIN_BLOB_AREA],
            reverse=True)

        bench_r, bench_area, has_circles, has_rects = compute_benchmarks(blobs)
        print(f"\n[{filename}]  bench_r={bench_r:.1f}  bench_area={bench_area:.0f}  "
              f"has_circles={has_circles}  has_rects={has_rects}")

        fig, ax = plt.subplots(figsize=(11, 13))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        counts = {k: 0 for k in COLORS}

        for area, cnt in blobs:
            shape, state, m = classify(cnt, bench_r, bench_area,
                                       has_circles, has_rects, img.shape)
            draw_ghost(ax, cnt, shape, bench_r, bench_area)
            annotate(ax, cnt, shape, state, m)
            counts[(state, shape)] += 1
            pr = m['perp_ratio']
            print(f"  {state:6} {shape:9}  af={m['af']:.3f}  sol={m['sol']:.3f}  "
                  f"r_ratio={m['r_ratio']:.3f}  size={m['size']:.2f}"
                  + (f"  perp={pr:.2f}" if pr is not None else ""))

        legend = [mpatches.Patch(color=v, label=f"{s} {sh}")
                  for (s, sh), v in COLORS.items()]
        ax.legend(handles=legend, loc='lower right', fontsize=8,
                  framealpha=0.85, title="Legend")
        summary = "  |  ".join(f"{s} {sh}: {n}"
                               for (s, sh), n in counts.items() if n > 0)
        plt.title(f"{filename}  ({time.time()-t0:.1f}s)\n{summary}", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()