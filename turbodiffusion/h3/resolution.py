"""Which canvas a request generates on, and which size it is handed over at.

H3-Base is a 768p model. The rule the checkpoint was released under anchors the
*short edge* at 768 and caps area at ``768*1344``, so its canvases are deliberately
not equal-area: 16:9 gets 1344x768 while 1:1 gets 768x768, a third fewer tokens for
the square one.

The tiers here anchor on area instead — ``720p`` means ``w*h`` near ``960**2``
whatever the ratio — which is the convention the rest of our stack states resolution
in. That makes every tier off-distribution for H3, and in both directions: below
native at wide ratios, above it at square ones, where ``720p`` resolves to 960x960
against a native 768x768. Nothing here clamps that. A tier is a request to run the
model on a canvas it was not trained on, and whether a given cell holds up is a
question for the output, not for this module.

Every canvas side is a multiple of 32, because the VAE compresses space 16x and the
DiT patch is 2x2, so an odd latent side has no whole patch to sit in. This is why a
literal 720-pixel side does not exist here: 720 is 22.5 patches, and a "720p" 16:9
canvas is 1312x736. The two sides are chosen together against the ratio rather than
resolved one at a time, which is what keeps that substitution close enough to the
requested ratio to matter — see :func:`_fit_canvas`.

The delivery size is the other half of the same question. Rounding the canvas to 32
throws away the exact ratio of the image that produced it — a 640x702 keyframe
generates on 768x832, which is 1.2% taller in proportion than it should be. The
delivery size covers the same area resolved to a multiple of 2 instead, which does
carry the ratio, and the result is resized onto it once at the end.
"""

from __future__ import annotations

import math

# H3's own canvas rule: short edge here, area capped there, both sides then rounded
# to 32. This is the geometry the checkpoint was released under.
H3_CANVAS_SHORT_EDGE = 768
H3_CANVAS_MAX_PIXELS = 768 * 1344

# Tier name → the edge of the square whose area the canvas matches. ``480p`` is not
# in the reference table this follows; it continues the same 4:3 relation the others
# hold (``540 * 4/3 = 720``, ``720 * 4/3 = 960``).
H3_RESOLUTION_PRESETS: dict[str, float] = {
    "360p": 480.0,
    "480p": 640.0,
    "540p": 720.0,
    "720p": 960.0,
}
# The tier the checkpoint was released at. Resolved by H3's own short-edge rule
# rather than by area, which is why it is not a row in the table above.
H3_NATIVE_RESOLUTION = "768p"
H3_CANVAS_MULTIPLE = 32
H3_DELIVERY_MULTIPLE = 2
# How far off the tier's area a source may be and still be delivered at its own size
# rather than a computed one, so an input that already fits is not resampled.
_DELIVER_AS_IS = (0.95, 1.05)
# How much of the tier's area a canvas may spend. Rounding to 32 cannot hit the area
# exactly, and the band is what buys the search room to hit the *ratio* instead.
_CANVAS_AREA_BAND = (0.90, 1.05)


def normalize_resolution(resolution: str | None) -> str | None:
    """A tier name, or ``None`` for H3's own short-edge rule.

    ``"768p"`` normalizes to ``None`` rather than to itself: native is not an area
    tier, it is the absence of one.
    """
    if resolution is None:
        return None
    name = str(resolution).strip().lower()
    if not name or name == H3_NATIVE_RESOLUTION:
        return None
    if name not in H3_RESOLUTION_PRESETS:
        raise ValueError(
            f"resolution must be one of {sorted(H3_RESOLUTION_PRESETS)} or "
            f"{H3_NATIVE_RESOLUTION!r}, got {resolution!r}"
        )
    return name


def parse_aspect(aspect_ratio: str) -> tuple[float, float]:
    """``"16:9"`` or ``"1619:1080"`` → the two numbers. Only their ratio is used.

    Rejects here what the canvas would reject later, so a request carrying a ratio H3
    cannot generate fails while it is still a request.
    """
    try:
        w, h = str(aspect_ratio).split(":")
        aspect_w, aspect_h = float(w), float(h)
    except ValueError as exc:
        raise ValueError(f"aspect ratio must look like '16:9', got {aspect_ratio!r}") from exc
    _check_ratio(aspect_w, aspect_h)
    return aspect_w, aspect_h


def aspect_string(width: float, height: float) -> str:
    """A size → the aspect string the rest of the pipeline passes around."""
    return f"{int(round(width))}:{int(round(height))}"


def _check_ratio(aspect_w: float, aspect_h: float) -> float:
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_w}:{aspect_h}")
    ratio = aspect_w / aspect_h
    if not 0.25 <= ratio <= 4.0:
        raise ValueError(
            f"H3 supports aspect ratios from 1:4 to 4:1, got {aspect_w}:{aspect_h} ({ratio:g})"
        )
    return ratio


def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: int = H3_CANVAS_MAX_PIXELS,
    multiple: int = H3_CANVAS_MULTIPLE,
) -> tuple[int, int]:
    """Resolve a display aspect ratio into an H3 canvas → ``(pixel_h, pixel_w)``.

    The short edge starts at ``short_edge``, the area is capped at ``max_pixels`` and
    both axes are then rounded to the nearest ``multiple``, so the final area can end up
    slightly above the pre-rounding budget. Only the ratio of the first two arguments
    matters. This is also the rule a reference *video* is put on, against its own aspect
    ratio — unlike a reference image, which carries no area budget.
    """
    ratio = _check_ratio(aspect_width, aspect_height)
    if ratio >= 1.0:
        pixel_w, pixel_h = short_edge * ratio, float(short_edge)
    else:
        pixel_w, pixel_h = float(short_edge), short_edge / ratio
    area = pixel_w * pixel_h
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        pixel_w, pixel_h = pixel_w * scale, pixel_h * scale
    return _round_multiple(pixel_h, multiple), _round_multiple(pixel_w, multiple)


def _floor_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(value // multiple) * multiple)


def _round_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def native_max_pixels(short_edge: int) -> int:
    """The area budget that goes with a short edge under H3's own rule.

    768 gets H3's ``768*1344`` exactly. A sweep that raises the short edge gets a
    proportionally wider canvas instead of being squeezed by a budget written for a
    smaller one.
    """
    return short_edge * int(round(short_edge * 1344 / H3_CANVAS_SHORT_EDGE))


def canvas_area(
    aspect_w: float,
    aspect_h: float,
    *,
    resolution: str | None = None,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: int | None = None,
) -> float:
    """How many pixels the tier spends on this ratio, before any rounding."""
    name = normalize_resolution(resolution)
    if name is not None:
        return H3_RESOLUTION_PRESETS[name] ** 2
    pixel_h, pixel_w = model_canvas(
        aspect_w, aspect_h, short_edge=short_edge, max_pixels=max_pixels
    )
    return float(pixel_h * pixel_w)


def model_canvas(
    aspect_w: float,
    aspect_h: float,
    *,
    resolution: str | None = None,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """The canvas the model generates on → ``(pixel_h, pixel_w)``, both /32."""
    name = normalize_resolution(resolution)
    if name is None:
        return resolve_canvas_size(
            aspect_w,
            aspect_h,
            short_edge=short_edge,
            max_pixels=max_pixels if max_pixels is not None else native_max_pixels(short_edge),
        )
    ratio = _check_ratio(aspect_w, aspect_h)
    return _fit_canvas(ratio, H3_RESOLUTION_PRESETS[name] ** 2)


def _fit_canvas(ratio: float, target_area: float) -> tuple[int, int]:
    """The /32 canvas nearest ``ratio`` among those spending about ``target_area``.

    Resolving each side against the multiple on its own costs up to 31 pixels there,
    which at 360p is 6% of the ratio — past what a keyframe derived from that same
    ratio is allowed to differ by, so the frame gets cropped to a canvas that was
    supposed to be its own. Choosing the two sides together holds the error near 2%
    at every tier, at the cost of an area that lands within a few percent of the
    tier's rather than always under it.
    """
    base_w = math.sqrt(target_area * ratio)
    lo, hi = target_area * _CANVAS_AREA_BAND[0], target_area * _CANVAS_AREA_BAND[1]
    best: tuple[tuple[float, float], tuple[int, int]] | None = None
    for i in range(-2, 3):
        pixel_w = _floor_multiple(base_w, H3_CANVAS_MULTIPLE) + i * H3_CANVAS_MULTIPLE
        if pixel_w < H3_CANVAS_MULTIPLE:
            continue
        for j in range(-2, 3):
            pixel_h = (
                _floor_multiple(pixel_w / ratio, H3_CANVAS_MULTIPLE)
                + j * H3_CANVAS_MULTIPLE
            )
            if pixel_h < H3_CANVAS_MULTIPLE:
                continue
            area = pixel_w * pixel_h
            if not lo <= area <= hi:
                continue
            key = (abs(math.log(pixel_w / pixel_h / ratio)), abs(area - target_area))
            if best is None or key < best[0]:
                best = (key, (pixel_h, pixel_w))
    if best is None:
        # Ratios extreme enough that no /32 pair lands in the band; spend under it.
        return (
            _floor_multiple(base_w / ratio, H3_CANVAS_MULTIPLE),
            _floor_multiple(base_w, H3_CANVAS_MULTIPLE),
        )
    return best[1]


def delivery_size(
    width: float,
    height: float,
    *,
    resolution: str | None = None,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """The size the result is handed over at → ``(pixel_h, pixel_w)``, both even.

    Same area as :func:`model_canvas` for the same request, resolved against a
    multiple of 2 instead of 32, so it keeps the ratio of ``width x height`` that
    the canvas had to round away.
    """
    _check_ratio(width, height)
    area = canvas_area(
        width, height, resolution=resolution, short_edge=short_edge, max_pixels=max_pixels
    )
    scale = math.sqrt(width * height / area)
    if _DELIVER_AS_IS[0] <= scale <= _DELIVER_AS_IS[1]:
        # Already the right size. Handing back its own dimensions avoids a resample
        # that would only move every pixel by less than half of one.
        return (
            _round_multiple(height, H3_DELIVERY_MULTIPLE),
            _round_multiple(width, H3_DELIVERY_MULTIPLE),
        )
    return (
        _round_multiple(height / scale, H3_DELIVERY_MULTIPLE),
        _round_multiple(width / scale, H3_DELIVERY_MULTIPLE),
    )


def keyframe_fit(
    src_w: float, src_h: float, target_w: float, target_h: float, tolerance: float = 0.05
) -> str:
    """How a keyframe is put on the canvas: ``"stretch"``, or ``"crop"`` to it first.

    Measured against the ratio the canvas was *asked* for, not the canvas itself. The
    canvas is a multiple of 32 and the request is not, and at 360p that gap alone can
    reach 6% — enough to crop the very frame the canvas was derived from. Comparing
    against the request instead leaves the rounding to :func:`delivery_size`, which is
    what undoes it, and leaves this deciding only the question it is for.

    That question is the closing frame. Stretching one shot at a genuinely different
    ratio than the opening frame would hand the model a distorted version of the frame
    the video has to end on, so it is cropped and the video ends on its middle.
    """
    ratio = (src_w / src_h) / (target_w / target_h)
    return "stretch" if 1 / (1 + tolerance) < ratio < 1 + tolerance else "crop"


def describe(
    aspect_w: float,
    aspect_h: float,
    *,
    resolution: str | None = None,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
) -> str:
    """One line for a log: which tier, which canvas, which delivery size."""
    ph, pw = model_canvas(aspect_w, aspect_h, resolution=resolution, short_edge=short_edge)
    dh, dw = delivery_size(aspect_w, aspect_h, resolution=resolution, short_edge=short_edge)
    tier = normalize_resolution(resolution) or H3_NATIVE_RESOLUTION
    deliver = "" if (dh, dw) == (ph, pw) else f" -> deliver {dw}x{dh}"
    return f"{tier} canvas {pw}x{ph}{deliver}"
