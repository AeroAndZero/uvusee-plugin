"""The geometry: which cube side each polygon gets, and where its UVs land in
a cubemap image. Also used by capture.py so the views we send to the server
line up exactly with how the result is projected back.

This file deliberately does not import bpy. It only needs numpy, so the maths
can be read, tested and reused on its own.
"""

import numpy as np

SIDES = ("FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM")

# "Projection space" is Blender's convention: Z is up and the FRONT image lands
# on the -Y side of the object (the side you see in Front view, numpad 1).
#
# For every side: (outward normal, image-right direction, image-up direction),
# as seen by someone standing outside the object looking at that side. These
# are exactly the directions you get by folding a cross-shaped cubemap into a
# box with the picture on the outside.
SIDE_FRAMES = {
    "FRONT":  ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "RIGHT":  ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "BACK":   ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "LEFT":   ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "TOP":    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "BOTTOM": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
}

# Dominant axis (0=X, 1=Y, 2=Z) and its sign -> side.
_AXIS_TO_SIDE = {
    (0, True): "RIGHT", (0, False): "LEFT",
    (1, True): "BACK", (1, False): "FRONT",
    (2, True): "TOP", (2, False): "BOTTOM",
}

# Image layouts: (columns, rows, {side: (column, row, quarter_turns)}).
# Row 0 is the top row of the image. quarter_turns says how many times the
# face is stored rotated 90 degrees counter-clockwise compared with the cross.
LAYOUTS = {
    # Horizontal cross (4x3), the most common, and the one sent to the server:
    #        TOP
    #  LEFT FRONT RIGHT BACK
    #       BOTTOM
    "CROSS_H": (4, 3, {
        "TOP": (1, 0, 0),
        "LEFT": (0, 1, 0), "FRONT": (1, 1, 0), "RIGHT": (2, 1, 0), "BACK": (3, 1, 0),
        "BOTTOM": (1, 2, 0),
    }),
    # Vertical cross (3x4). BACK hangs under BOTTOM, so it is stored upside down.
    "CROSS_V": (3, 4, {
        "TOP": (1, 0, 0),
        "LEFT": (0, 1, 0), "FRONT": (1, 1, 0), "RIGHT": (2, 1, 0),
        "BOTTOM": (1, 2, 0),
        "BACK": (1, 3, 2),
    }),
    # Strips and the 3x2 grid use the usual +X -X +Y -Y +Z -Z order
    # (right, left, top, bottom, front, back).
    "STRIP_H": (6, 1, {
        "RIGHT": (0, 0, 0), "LEFT": (1, 0, 0), "TOP": (2, 0, 0),
        "BOTTOM": (3, 0, 0), "FRONT": (4, 0, 0), "BACK": (5, 0, 0),
    }),
    "STRIP_V": (1, 6, {
        "RIGHT": (0, 0, 0), "LEFT": (0, 1, 0), "TOP": (0, 2, 0),
        "BOTTOM": (0, 3, 0), "FRONT": (0, 4, 0), "BACK": (0, 5, 0),
    }),
    "GRID_3X2": (3, 2, {
        "RIGHT": (0, 0, 0), "LEFT": (1, 0, 0), "TOP": (2, 0, 0),
        "BOTTOM": (0, 1, 0), "FRONT": (1, 1, 0), "BACK": (2, 1, 0),
    }),
}

LAYOUT_NAMES = {
    "CROSS_H": "Horizontal Cross",
    "CROSS_V": "Vertical Cross",
    "STRIP_H": "Horizontal Strip",
    "STRIP_V": "Vertical Strip",
    "GRID_3X2": "3×2 Grid",
}

# Rotations (about Z) that take object space to projection space, i.e. bring
# the chosen "front" axis onto -Y.
FRONT_AXIS_MATRICES = {
    "NEG_Y": np.eye(3),
    "POS_Y": np.diag([-1.0, -1.0, 1.0]),
    "POS_X": np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    "NEG_X": np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}


def layout_rects(layout):
    """Where each side sits in the image, as UV rectangles:
    {side: (u0, v0, du, dv, quarter_turns)} with (u0, v0) the bottom-left corner.

    `layout` is a key of LAYOUTS, or already such a dict (e.g. from
    net_uv_rects) and is then returned unchanged.
    """
    if isinstance(layout, dict):
        return dict(layout)
    cols, rows, cells = LAYOUTS[layout]
    return {side: (col / cols, (rows - 1 - row) / rows, 1.0 / cols, 1.0 / rows, turns)
            for side, (col, row, turns) in cells.items()}


def grid_pixel_rects(layout, face_px):
    """Pixel rectangles (x, y, w, h), top-left origin, of a grid layout."""
    cols, rows, cells = LAYOUTS[layout]
    return (cols * face_px, rows * face_px,
            {side: (col * face_px, row * face_px, face_px, face_px)
             for side, (col, row, _turns) in cells.items()})


# Which box dimensions each side shows, as (across, up) in its upright view:
# x = width (left-right), y = depth (front-back), z = height.
SIDE_DIMS = {"FRONT": (0, 2), "BACK": (0, 2), "LEFT": (1, 2), "RIGHT": (1, 2),
             "TOP": (0, 1), "BOTTOM": (0, 1)}

LAYOUT_TITLES = {          # names used in the server request
    "CROSS_H": "horizontal_cross", "CROSS_V": "vertical_cross",
    "STRIP_H": "horizontal_strip", "STRIP_V": "vertical_strip", "GRID_3X2": "grid_3x2",
}


def proportional_layout(layout, extent_x, extent_y, extent_z, budget_px, max_side,
                        min_px=8, multiple=16):
    """A grid layout (a LAYOUTS key) with every side at its true proportions
    and the same pixel density ("Keep Proportions").

    Each column is as wide as its widest side and each row as tall as its
    tallest; sides are centred in their slot. For the crosses this is exactly
    the box unfolded, e.g. the horizontal cross:

                [ TOP    x*y ]
        [LEFT y*z][FRONT x*z][RIGHT y*z][BACK x*z]
                [ BOTTOM x*y ]

    The scale is chosen so the image has about `budget_px` pixels and no side
    longer than `max_side`. The total size is rounded up to a multiple of
    `multiple` (friendlier for image models); the extra is left empty.

    Returns (width, height, {side: (x, y, w, h)}) in pixels, top-left origin,
    where (w, h) is the side as stored (i.e. after its quarter turns).
    """
    cols, rows, cells = LAYOUTS[layout]
    ext = [max(float(e), 0.0) for e in (extent_x, extent_y, extent_z)]
    biggest = max(max(ext), 1e-9)
    # Keep degenerate (flat) sides at a few pixels rather than zero.
    ext = [max(e, biggest * 1e-3) for e in ext]

    def dims(side):
        a, b = SIDE_DIMS[side]
        w, h = ext[a], ext[b]
        return (h, w) if cells[side][2] % 2 else (w, h)

    col_units = [max([dims(sd)[0] for sd, c in cells.items() if c[0] == i] or [0]) for i in range(cols)]
    row_units = [max([dims(sd)[1] for sd, c in cells.items() if c[1] == j] or [0]) for j in range(rows)]
    total_w, total_h = sum(col_units), sum(row_units)
    scale = min((budget_px / (total_w * total_h)) ** 0.5, max_side / max(total_w, total_h))

    def px(v):
        return max(min_px, int(round(v * scale)))

    side_px = {sd: (px(dims(sd)[0]), px(dims(sd)[1])) for sd in cells}
    col_w = [max(side_px[sd][0] for sd, c in cells.items() if c[0] == i) for i in range(cols)]
    row_h = [max(side_px[sd][1] for sd, c in cells.items() if c[1] == j) for j in range(rows)]
    col_x = [sum(col_w[:i]) for i in range(cols)]
    row_y = [sum(row_h[:j]) for j in range(rows)]
    rects = {}
    for sd, (c, r, _turns) in cells.items():
        w, h = side_px[sd]
        rects[sd] = (col_x[c] + (col_w[c] - w) // 2, row_y[r] + (row_h[r] - h) // 2, w, h)

    def up(v):
        return -(-v // multiple) * multiple
    return up(sum(col_w)), up(sum(row_h)), rects


def proportional_net(extent_x, extent_y, extent_z, budget_px, max_side, min_px=8, multiple=16):
    """The horizontal cross at true proportions (see proportional_layout)."""
    return proportional_layout("CROSS_H", extent_x, extent_y, extent_z, budget_px, max_side,
                               min_px, multiple)


def net_uv_rects(width, height, pixel_rects, turns=None):
    """Pixel rectangles (top-left origin) -> UV rectangles for compute_uvs."""
    turns = turns or {}
    return {side: (x / width, 1.0 - (y + h) / height, w / width, h / height, turns.get(side, 0))
            for side, (x, y, w, h) in pixel_rects.items()}


def detect_layout(width, height, tolerance=0.02):
    """Guess the layout from the image's aspect ratio. Returns a key or None."""
    if not width or not height:
        return None
    ratio = width / height
    for key, (cols, rows, _cells) in LAYOUTS.items():
        expected = cols / rows
        if abs(ratio - expected) / expected < tolerance:
            return key
    return None


def looks_equirectangular(width, height):
    return bool(width and height) and abs(width / height - 2.0) < 0.04


def compute_bounds(points, front_axis="NEG_Y", uniform=False, flat_fix=True):
    """Centre and half-size of the points' bounding box, in projection space.

    The capture and the projection must use the very same box, otherwise the
    returned texture would land slightly off. Both call this function.
    """
    rot = FRONT_AXIS_MATRICES[front_axis]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3) @ rot.T
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if uniform:
        half[:] = half.max()          # same scale on every axis
    if flat_fix:
        biggest = half.max()
        # Flat objects (e.g. a plane): give the flat axis a sensible size so the
        # side views aren't infinitely thin. No polygon faces those sides anyway.
        half[half < 1e-9] = biggest if biggest > 1e-9 else 1.0
    return mid, half


def compute_uvs(co, poly_normals, poly_centers, loop_verts, loop_poly, *,
                layout, mode="BOX", front_axis="NEG_Y", uniform=False,
                padding_px=1.0, image_size=(1024, 768), bounds=None,
                rotations=None, flips=None, swap_front_back=False):
    """Work out a UV for every face corner (loop) of a mesh.

    co           (V, 3) vertex positions in object space
    poly_normals (P, 3) polygon normals
    poly_centers (P, 3) polygon centres
    loop_verts   (L,)   vertex index of each loop
    loop_poly    (L,)   polygon index of each loop
    layout       a LAYOUTS key, or UV rectangles from net_uv_rects()
    mode         "BOX": each polygon takes the side its normal faces most,
                        and is projected straight (orthographically) onto it.
                 "SPHERE": each polygon takes the side its direction from the
                        object's centre points to, and is projected through the
                        centre, the way a real cubemap is sampled. Seamless on
                        round objects.
    bounds       (mid, half) from compute_bounds(); computed from co if None.
    rotations / flips: optional per-side fixes {side: quarter_turns / bool}.

    Returns (uvs (L, 2) float32, poly_side (P,) indices into SIDES).
    """
    rotations = rotations or {}
    flips = flips or {}
    rects = layout_rects(layout)
    if swap_front_back:
        rects["FRONT"], rects["BACK"] = rects["BACK"], rects["FRONT"]

    # 1. Move everything into projection space and normalise the bounding box
    #    to the cube [-1, 1]^3, so each side's image covers that whole side.
    rot = FRONT_AXIS_MATRICES[front_axis]
    if bounds is None:
        bounds = compute_bounds(co, front_axis, uniform)
    mid, half = bounds
    co = np.asarray(co, dtype=np.float64) @ rot.T
    normals = np.asarray(poly_normals, dtype=np.float64) @ rot.T
    centers = np.asarray(poly_centers, dtype=np.float64) @ rot.T
    p = (co - mid) / half

    # 2. Pick a side for every polygon from its dominant axis.
    pick = normals if mode == "BOX" else (centers - mid) / half
    axis = np.abs(pick).argmax(axis=1)
    positive = pick[np.arange(len(pick)), axis] >= 0
    side_lookup = np.array([[SIDES.index(_AXIS_TO_SIDE[(a, s)]) for s in (False, True)]
                            for a in range(3)])
    poly_side = side_lookup[axis, positive.astype(int)]

    # 3. Per face corner: project onto its side, then place it in the image.
    loop_side = poly_side[loop_poly]
    pl = p[loop_verts]
    uvs = np.zeros((len(loop_verts), 2), dtype=np.float64)

    width, height = image_size

    for index, side in enumerate(SIDES):
        mask = loop_side == index
        if not mask.any():
            continue
        normal, right, up = (np.array(v, dtype=np.float64) for v in SIDE_FRAMES[side])
        pts = pl[mask]
        a = pts @ right
        b = pts @ up
        if mode == "SPHERE":
            depth = np.maximum(pts @ normal, 1e-6)
            a = a / depth
            b = b / depth
        # Stay inside this face's square even for points slightly outside the
        # box (e.g. a cage vertex of a Subdivision Surface object).
        a = np.clip(a, -1.0, 1.0)
        b = np.clip(b, -1.0, 1.0)

        # Position within this face's square, 0..1 with (0, 0) bottom-left.
        u = (a + 1.0) / 2.0
        v = (b + 1.0) / 2.0
        if flips.get(side):
            u = 1.0 - u

        u0, v0, du, dv, turns = rects[side]
        for _ in range((turns + rotations.get(side, 0)) % 4):
            u, v = 1.0 - v, u                 # rotate 90 degrees counter-clockwise

        # Shrink slightly so texture filtering never samples the neighbour cell.
        pad_u = min(0.49, padding_px / (du * width)) if width else 0.0
        pad_v = min(0.49, padding_px / (dv * height)) if height else 0.0
        u = pad_u + u * (1.0 - 2.0 * pad_u)
        v = pad_v + v * (1.0 - 2.0 * pad_v)

        uvs[mask, 0] = u0 + u * du
        uvs[mask, 1] = v0 + v * dv

    return uvs.astype(np.float32), poly_side