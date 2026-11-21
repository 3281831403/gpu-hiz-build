"""Hi-Z 遮挡剔除的行为契约测试（初始全部失败）。"""

import pytest

from hiz.buffer import (
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


def test_build_level_uses_max_depth() -> None:
    """2x2 归约必须取最远深度，而不是平均值。"""
    prev = [[0.10, 0.90], [0.80, 0.20]]

    assert build_level(prev) == [[0.90]]


def test_build_level_keeps_odd_right_and_bottom_edges() -> None:
    """奇数尺寸要向上取整，最右列和最下行不能丢。"""
    prev = [
        [0.10, 0.20, 0.90],
        [0.30, 0.40, 0.70],
        [0.60, 0.80, 0.95],
    ]

    level = build_level(prev)

    assert len(level) == 2
    assert all(len(row) == 2 for row in level)
    assert level == [[0.40, 0.90], [0.80, 0.95]]


def test_build_mips_strictly_shrinks_to_one_by_one() -> None:
    """层级必须持续缩小，不能停在 2x2，最后一层应是 1x1。"""
    depth = [[(y * 8 + x) / 63.0 for x in range(8)] for y in range(8)]

    mips = build_mips(depth, 4)

    assert [len(level) for level in mips] == [8, 4, 2, 1]
    assert [len(level[0]) for level in mips] == [8, 4, 2, 1]
    assert mips[0] == depth
    assert mips[-1] == [[1.0]]


def test_pick_level_rounds_up_for_coverage() -> None:
    """3x2 矩形需要 4 像素跨度，必须选第 2 层而不是第 1 层。"""
    rect = {"x": 4, "y": 7, "w": 3, "h": 2}

    assert pick_level(rect, 4) == 2
    assert pick_level({"x": 0, "y": 0, "w": 1, "h": 1}, 4) == 0


def test_occluded_is_conservative_at_boundary_and_coplanar() -> None:
    """只读取矩形覆盖的像素；贴面深度在容差内也算遮挡。"""
    mips = [[[0.10, 0.90], [0.90, 0.90]]]
    rect = {"x": 0, "y": 0, "w": 1, "h": 1}

    # 只覆盖左上角 0.10；若把右边界算进去会错误读到 0.90。
    assert occluded(rect, 0, mips, 0.10) is True
    # 比块内最远值近 5e-7，仍在 1e-6 容差内。
    assert occluded(rect, 0, mips, 0.10 - 5e-7) is True
    # 近 2e-6，已超出容差，必须保留可见。
    assert occluded(rect, 0, mips, 0.10 - 2e-6) is False


def test_clamp_rect_crops_without_reading_outside_image() -> None:
    """越界矩形必须裁剪到图像内，完全在外时宽高为 0。"""
    assert clamp_rect({"x": -2, "y": -1, "w": 5, "h": 4}, 5, 5) == {
        "x": 0,
        "y": 0,
        "w": 3,
        "h": 3,
    }
    assert clamp_rect({"x": 10, "y": 10, "w": 2, "h": 2}, 5, 5) == {
        "x": 5,
        "y": 5,
        "w": 0,
        "h": 0,
    }


def test_merge_rects_uses_union_for_overlapping_queries() -> None:
    """重叠矩形应合并成包围盒，不能取交集。"""
    rects = [
        {"x": 0, "y": 0, "w": 4, "h": 4},
        {"x": 3, "y": 2, "w": 4, "h": 4},
    ]

    assert merge_rects(rects) == [{"x": 0, "y": 0, "w": 7, "h": 6}]


def test_validate_mips_reports_bad_depths_and_layers() -> None:
    """越界、NaN、空层和尺寸未递减都要进入问题列表。"""
    assert validate_mips([[[0.25, 0.50]]]) == []
    assert "depth_out_of_range" in validate_mips([[[1.20]]])
    assert "depth_out_of_range" in validate_mips([[[float("nan")]]])
    assert "depth_out_of_range" in validate_mips([[[float("inf")]]])
    assert "empty_level" in validate_mips([[]])
    assert "size_not_decreasing" in validate_mips([[[0.10]], [[0.20]]])


def test_hiz_batch_isolates_frames_and_rolls_back() -> None:
    """每帧有自己的 mip 链；任一帧非法时整体失败且不改输入。"""
    frame_a = [[0.10, 0.20], [0.30, 0.40]]
    frame_b = [[0.90, 0.80], [0.70, 0.60]]
    snapshot_a = [row[:] for row in frame_a]
    snapshot_b = [row[:] for row in frame_b]

    result = hiz_batch([frame_a, frame_b], {"levels": 2})
    chains = result["mips"]

    assert result["count"] == 2
    assert len(chains) == 2
    assert chains[0] is not chains[1]
    assert chains[0][0] == frame_a
    assert chains[1][0] == frame_b
    assert chains[0][-1] == [[0.40]]
    assert chains[1][-1] == [[0.90]]

    bad = [row[:] for row in frame_b]
    bad[0][0] = float("nan")
    with pytest.raises(HizError):
        hiz_batch([frame_a, bad], {"levels": 2})

    assert frame_a == snapshot_a
    assert frame_b == snapshot_b


def test_hiz_batch_rejects_nonpositive_levels() -> None:
    """批量配置的层数上限必须为正，否则抛 HizError。"""
    with pytest.raises(HizError):
        hiz_batch([[[0.10]]], {"levels": 0})
    with pytest.raises(HizError):
        hiz_batch([[[0.10]]], {"levels": -2})


def test_single_pixel_map_has_exactly_one_level() -> None:
    """1x1 图无法再归约，即使允许更多层也只有一层。"""
    assert build_mips([[0.35]], 8) == [[[0.35]]]


def test_all_far_depth_keeps_rectangles_visible() -> None:
    """深度图全为 1.0 时，较近的任意查询矩形都不应被遮挡。"""
    depth = [[1.0] * 4 for _ in range(4)]
    mips = build_mips(depth, 3)

    assert occluded({"x": 0, "y": 0, "w": 4, "h": 4}, 0, mips, 0.25) is False
    assert occluded({"x": 3, "y": 2, "w": 1, "h": 2}, 1, mips, 0.25) is False


def test_build_mips_rejects_nonpositive_levels() -> None:
    """层数 0 或负数不能进入构建循环，必须抛 HizError。"""
    with pytest.raises(HizError):
        build_mips([[0.20]], 0)
    with pytest.raises(HizError):
        build_mips([[0.20]], -3)


def test_empty_depth_map_builds_no_levels() -> None:
    """空图没有第 0 层，直接返回空列表。"""
    assert build_mips([], 4) == []


def test_hiz_batch_success_preserves_frame_order() -> None:
    """批量成功时结果链的顺序必须与输入帧顺序一致。"""
    frame_a = [[0.10, 0.20], [0.30, 0.40]]
    frame_b = [[0.90, 0.80], [0.70, 0.60]]

    result = hiz_batch([frame_a, frame_b], {"levels": 2})

    assert result["count"] == 2
    assert result["mips"][0][0] == frame_a
    assert result["mips"][1][0] == frame_b
    assert result["mips"][0][-1] == [[0.40]]
    assert result["mips"][1][-1] == [[0.90]]
