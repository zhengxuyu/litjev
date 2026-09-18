"""Deterministic engine-assisted descriptions, not RGB perception or an expert policy."""

import numpy as np

OBSERVATION_KIND = "engine_labels_depth_text"
MAX_OBJECTS = 32
SECTORS = ("left", "center", "right")


def describe_buffers(objects, labels, depth):
    if labels is None or depth is None or labels.shape != depth.shape:
        raise ValueError("Text observations require matching label and depth buffers")
    height, width = labels.shape
    lines = [
        "Engine-assisted visible-scene description (not RGB recognition).",
        "Depth values are raw 8-bit buffer values, not meters.",
        "Horizontal x runs from -1 (left) through 0 (crosshair) to +1 (right).",
    ]
    visible = []
    for obj in objects or []:
        mask = labels == obj.value
        _, xs = np.nonzero(mask)
        if not len(xs):
            continue
        x = float((xs.min() + xs.max() + 1) / width - 1)
        side = "left" if x < -0.12 else "right" if x > 0.12 else "center"
        visible.append((len(xs), (
            f"{obj.object_name}: {side}, x={x:+.3f}, "
            f"visible_area={len(xs) / (width * height):.4f}, "
            f"median_depth_raw={float(np.median(depth[mask])):.1f}."
        )))
    visible.sort(key=lambda item: (-item[0], item[1]))
    lines.append(f"Visible labeled objects: {len(visible)}.")
    lines.extend(text for _, text in visible[:MAX_OBJECTS])
    if len(visible) > MAX_OBJECTS:
        lines.append(f"Omitted {len(visible) - MAX_OBJECTS} smaller visible objects.")
    band = depth[height // 3:2 * height // 3]
    lines.append("Middle-screen depth by sector (not a traversability guarantee):")
    for name, values in zip(SECTORS, np.array_split(band, 3, axis=1), strict=True):
        lines.append(f"{name}: median_depth_raw={float(np.median(values)):.1f}.")
    return "\n".join(lines)
