"""Blend & Bake: turn the painted views into one UV texture.

Box projection gives every polygon the colours of ONE view, chosen by which way
it faces. That stretches slanted surfaces and makes hard seams. Baking instead:

  1. At capture time, each view is also rendered as a "lookup" image: for every
     pixel, the UV coordinate of the surface visible there, and how directly
     the view looks at that surface (cosine between surface normal and view
     direction). The renderer's depth test takes care of visibility.
  2. When the painted views come back, every painted pixel is added into the
     texture at its UV, weighted by cosine ** sharpness. Surfaces seen by
     several views get a smooth blend that favours the most direct view.
  3. Pixels at silhouettes and occlusion edges are skipped (image models smear
     colours there), and texels no view saw are filled from their surroundings
     ("push-pull" fill), which also bleeds colour past the UV island borders.

Pure numpy, no bpy: easy to test, and the same code works for any number of
views, so more cameras can be added later.
"""

import numpy as np

EDGE_FACTOR = 3.0     # a UV jump this many times the usual step counts as an edge


def bilinear(image, x, y):
    """Sample an (H, W, C) image (top row first) at float pixel coords x, y."""
    h, w = image.shape[:2]
    x = np.clip(x, 0.0, w - 1.0)
    y = np.clip(y, 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def typical_step(lookup):
    """Median UV distance between neighbouring on-object lookup pixels: how far
    apart this view's samples normally are in the texture."""
    u, v, alpha = lookup[..., 0], lookup[..., 1], lookup[..., 3]
    inside = alpha > 0.5
    steps = []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        both = inside[a] & inside[b]
        d = np.maximum(np.abs(u[a] - u[b]), np.abs(v[a] - v[b]))[both]
        if d.size:
            steps.append(d)
    return float(np.median(np.concatenate(steps))) if steps else 0.0


def usable_pixels(lookup, min_cos, jump):
    """Mask of lookup pixels worth baking: on the object, not too grazing, and
    not next to a silhouette or a jump in UV (occlusion edge or UV seam)."""
    u, v, cos, alpha = (lookup[..., i] for i in range(4))
    inside = alpha > 0.5
    ok = inside & (cos > min_cos)
    # Neighbours: any background or large UV jump around a pixel rejects it.
    padded_in = np.pad(inside, 1, constant_values=False)
    padded_u = np.pad(u, 1, mode="edge")
    padded_v = np.pad(v, 1, mode="edge")
    h, w = u.shape
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        n_in = padded_in[dy:dy + h, dx:dx + w]
        n_u = padded_u[dy:dy + h, dx:dx + w]
        n_v = padded_v[dy:dy + h, dx:dx + w]
        ok &= n_in & (np.abs(n_u - u) < jump) & (np.abs(n_v - v) < jump)
    return ok


def splat(size, tx, ty, colors, weights):
    """Add samples into a size x size texture with bilinear footprints.

    Returns (cnt, wsum, acc): per texel the summed footprint, the summed
    footprint * weight, and the summed footprint * weight * colour (T*T, 3).
    """
    n = size * size
    x0 = np.floor(tx).astype(np.int64)
    y0 = np.floor(ty).astype(np.int64)
    fx = (tx - x0).astype(np.float32)
    fy = (ty - y0).astype(np.float32)
    idx, foot, src = [], [], []
    for dx, dy, f in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx
        yi = y0 + dy
        keep = np.nonzero((xi >= 0) & (xi < size) & (yi >= 0) & (yi < size))[0]
        idx.append(yi[keep] * size + xi[keep])
        foot.append(f[keep])
        src.append(keep)
    # One pass per quantity over all four footprints together.
    idx = np.concatenate(idx)
    foot = np.concatenate(foot)
    src = np.concatenate(src)
    bw = foot * weights[src]
    cnt = np.bincount(idx, weights=foot, minlength=n)
    wsum = np.bincount(idx, weights=bw, minlength=n)
    acc = np.stack([np.bincount(idx, weights=bw * colors[src, c], minlength=n)
                    for c in range(3)], axis=1)
    return cnt, wsum, acc


def _box_sum(a, radius, axis):
    """Sum of each (2r+1)-long window along one axis (zero outside)."""
    k = 2 * radius + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    c = np.pad(a, pad).cumsum(axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, None)
    lo[axis] = slice(None, -k)
    return c[tuple(hi)] - c[tuple(lo)]


def box_fill(values, weight, radius):
    """Close small gaps: texels with zero weight get the weighted average of
    known texels within `radius` (a (2r+1)^2 box). Known texels are kept.

    values (T, T, C), weight (T, T). Returns (filled values, mask of texels
    that now have a value). Separable running sums: cheap, and short enough
    to stay accurate in single precision.
    """
    stack = np.concatenate([values * weight[..., None], weight[..., None]], axis=-1)
    box = _box_sum(_box_sum(stack, radius, 0), radius, 1)
    near_w = box[..., -1]
    near = box[..., :-1] / np.maximum(near_w, 1e-12)[..., None]
    known = weight > 0
    filled = np.where(known[..., None], values, near)
    return filled, known | (near_w > 1e-6)


def push_pull(values, weight, max_levels=None):
    """Fill texels with zero weight from their neighbourhood, at every scale.

    values: (T, T, C) per-texel values, weight: (T, T) confidence (0 = hole).
    Halve the resolution repeatedly (weighted average), then come back up,
    filling each hole from the coarser level. Known texels are kept exactly.
    max_levels limits how far holes are filled (2**max_levels texels); texels
    still empty come back with zero in the returned mask.

    Returns (filled values, mask of texels that now have a value).
    """
    levels = [(values * weight[..., None], weight)]
    while levels[-1][1].shape[0] > 1 and levels[-1][1].shape[1] > 1 and \
            (max_levels is None or len(levels) <= max_levels):
        c, w = levels[-1]
        h, wd = w.shape
        h2, w2 = h // 2 * 2, wd // 2 * 2
        c = c[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2, -1).sum(axis=(1, 3))
        w = w[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).sum(axis=(1, 3))
        levels.append((c, w))
    c, w = levels[-1]
    filled = c / np.maximum(w, 1e-12)[..., None]
    have = w > 0
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(filled, 2, axis=0), 2, axis=1)
        up_have = np.repeat(np.repeat(have, 2, axis=0), 2, axis=1)
        pad = ((0, c.shape[0] - up.shape[0]), (0, c.shape[1] - up.shape[1]))
        up = np.pad(up, pad + ((0, 0),), mode="edge")
        up_have = np.pad(up_have, pad, mode="edge")
        known = w > 0
        own = c / np.maximum(w, 1e-12)[..., None]
        filled = np.where(known[..., None], own, up)
        have = known | up_have
    return filled, have


def bake(painted, views, size=2048, sharpness=4.0, min_cos=0.17, close_gaps=2):
    """Bake painted views into a size x size UV texture.

    painted  (H, W, 3) float 0..1, top row first: the image that came back,
             at the capture's size.
    views    list of (rect, lookup): rect = (x, y, w, h) of the view in
             `painted` (top-left origin); lookup = (h', w', 4) float array
             (u, v, cos, alpha), top row first, covering that rect.
    close_gaps  each view first fills its own gaps up to this many texels away,
             so two sparse views don't interleave into stripes.
    Returns (texture (size, size, 3) top row first, fraction of texels painted
    directly by at least one view).
    """
    total = np.zeros((size, size, 3), dtype=np.float32)
    total_w = np.zeros((size, size), dtype=np.float32)
    painted_any = np.zeros((size, size), dtype=bool)
    for (x, y, w, h), lookup in views:
        lh, lw = lookup.shape[:2]
        # A UV jump much bigger than this view's usual pixel-to-pixel step is an
        # edge (silhouette, occlusion or UV seam). Views that sample the surface
        # coarsely have bigger usual steps, so the limit scales with them.
        jump = max(4.0 / size, EDGE_FACTOR * typical_step(lookup))
        ok = usable_pixels(lookup, min_cos, jump)
        if not ok.any():
            continue
        rows, cols = np.nonzero(ok)
        # Where this lookup pixel sits in the painted image.
        px = x + (cols + 0.5) * (w / lw) - 0.5
        py = y + (rows + 0.5) * (h / lh) - 0.5
        colors = bilinear(painted, px, py)
        u = lookup[rows, cols, 0]
        v = lookup[rows, cols, 1]
        weights = np.power(np.clip(lookup[rows, cols, 2], 0.0, 1.0), sharpness)
        # Texture rows go top-down, UV v goes bottom-up.
        cnt, wsum, acc = splat(size, u * size - 0.5, (1.0 - v) * size - 0.5, colors, weights)
        # This view alone: its colour and its (average) weight per texel, with
        # its own small gaps closed.
        color_v = (acc / np.maximum(wsum, 1e-12)[:, None]).reshape(size, size, 3)
        weight_v = (wsum / np.maximum(cnt, 1e-12)).reshape(size, size)
        cnt = cnt.reshape(size, size).astype(np.float32)
        both, have = box_fill(np.dstack([color_v, weight_v]).astype(np.float32), cnt, close_gaps)
        # Gap-filled texels count half: the view didn't actually see them.
        wv = np.where(cnt > 0, both[..., 3], np.where(have, 0.5 * both[..., 3], 0.0))
        total += both[..., :3] * wv[..., None]
        total_w += wv
        painted_any |= cnt > 0

    color = total / np.maximum(total_w, 1e-12)[..., None]
    texture, _have = push_pull(color, (total_w > 1e-12).astype(np.float32))
    return np.clip(texture, 0.0, 1.0), float(painted_any.mean())