"""Hi-Z（层次深度）遮挡剔除练习包。"""

from .buffer import (
    EPS,
    HizError,
    build_level,
    build_mips,
    clamp_rect,
    hiz_batch,
    merge_rects,
    occluded,
    pick_level,
    validate_mips,
)

__all__ = [
    "EPS",
    "HizError",
    "build_level",
    "build_mips",
    "pick_level",
    "occluded",
    "clamp_rect",
    "merge_rects",
    "validate_mips",
    "hiz_batch",
]
