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


def compute_bounds(points, front_axis="NEG_Y", uniform=False):
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
    cols, rows, cells = LAYOUTS[layout]
    cells = dict(cells)
    if swap_front_back:
        cells["FRONT"], cells["BACK"] = cells["BACK"], cells["FRONT"]

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
    pad_u = padding_px / (width / cols) if width else 0.0
    pad_v = padding_px / (height / rows) if height else 0.0

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

        col, row, turns = cells[side]
        for _ in range((turns + rotations.get(side, 0)) % 4):
            u, v = 1.0 - v, u                 # rotate 90 degrees counter-clockwise

        # Shrink slightly so texture filtering never samples the neighbour cell.
        u = pad_u + u * (1.0 - 2.0 * pad_u)
        v = pad_v + v * (1.0 - 2.0 * pad_v)

        # Image rows count from the top, UVs count from the bottom.
        uvs[mask, 0] = (col + u) / cols
        uvs[mask, 1] = (rows - 1 - row + v) / rows

    return uvs.astype(np.float32), poly_side
