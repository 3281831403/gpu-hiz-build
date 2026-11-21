"""Hi-Z 深度金字塔的构建、选层、遮挡测试与批量入口。

约定
----
* 深度图是二维 ``list[list[float]]``，``depth[y][x]`` 为像素深度；
  深度值域为 ``[0.0, 1.0]``，``0.0`` 最近、``1.0`` 最远。
* 矩形是 ``{"x": int, "y": int, "w": int, "h": int}``，采用左上角原点、
  半开区间 ``[x, x + w) x [y, y + h)``；未特别说明时 ``w``、``h`` 必须为正。
* 金字塔第 0 层是原图，第 ``l`` 层的一个像素对应第 0 层 ``2**l`` 见方的块。
* 所有会影响遮挡判定的浮点比较都使用 ``EPS = 1e-6`` 的容差。
* 参数非法或结构不合法时抛出 :class:`HizError`；``validate_mips`` 例外，
  它把问题作为字符串列表返回。
"""

from __future__ import annotations

import math

EPS = 1e-6


class HizError(Exception):
    """Hi-Z 金字塔的输入或状态不合法。"""


# ---------------------------------------------------------------------------
# 内部校验辅助
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    """是否为真正的整数（``bool`` 不算）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """是否为实数值（``bool`` 不算）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_valid_depth(value: object) -> bool:
    """深度是否为 ``[0.0, 1.0]`` 内的有限实数（拒绝 NaN / inf）。"""
    if not _is_real(value):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _check_level_shape(level: object) -> tuple[int, int]:
    """校验深度层是矩形的二维序列，返回 ``(高度, 宽度)``。

    空层返回 ``(0, 0)``；非矩形或元素不是行时抛出 :class:`HizError`。
    """
    if not isinstance(level, (list, tuple)):
        raise HizError("深度层必须是二维列表")
    if len(level) == 0:
        return 0, 0

    first = level[0]
    if not isinstance(first, (list, tuple)):
        raise HizError("深度层的每一行必须是列表")
    width = len(first)
    for row in level:
        if not isinstance(row, (list, tuple)):
            raise HizError("深度层的每一行必须是列表")
        if len(row) != width:
            raise HizError("深度图必须是矩形（各行长度一致）")
    return len(level), width


def _check_depths(level: list, context: str) -> None:
    """逐像素校验深度值域；任何违规都抛出 :class:`HizError`。"""
    for y, row in enumerate(level):
        for x, value in enumerate(row):
            if not _is_valid_depth(value):
                raise HizError(
                    f"{context} 的深度在 ({y}, {x}) 处非法: {value!r}"
                )


def _check_rect_fields(rect: object) -> tuple[int, int, int, int]:
    """取出并校验矩形字段，宽高可为 0，返回 ``(x, y, w, h)``。"""
    if not isinstance(rect, dict):
        raise HizError("矩形必须是字典")
    for key in ("x", "y", "w", "h"):
        if key not in rect:
            raise HizError(f"矩形缺少字段: {key}")

    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    if not (_is_int(x) and _is_int(y)):
        raise HizError("矩形坐标 x、y 必须是整数")
    if not (_is_int(w) and _is_int(h)):
        raise HizError("矩形宽高 w、h 必须是整数")
    if w < 0 or h < 0:
        raise HizError("矩形宽高不能为负")
    return x, y, w, h


def _check_positive_rect(rect: object) -> tuple[int, int, int, int]:
    """取出矩形字段，并要求宽高为正。"""
    x, y, w, h = _check_rect_fields(rect)
    if w <= 0 or h <= 0:
        raise HizError("矩形宽高必须为正")
    return x, y, w, h


def _copy_level(level: list) -> list:
    """复制一层深度图，保证返回值与原容器不共享行列表。"""
    return [list(row) for row in level]


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------


def build_level(prev: list[list[float]]) -> list[list[float]]:
    """把一层深度图按 2x2 归约成下一层。

    对每个 2x2 邻域取 **最大深度**（最远值）而不是平均值；这是保守的
    Hi-Z 表达，可保证块内没有比该值更远的几何。奇数宽高时向上取整，
    最右列和最下行不足 2 个像素时用现有像素参与取最大值，不能丢弃。
    输入为空时返回 ``[]``；输入必须是矩形，且所有深度有限并位于
    ``[0, 1]``。返回新列表，不修改 ``prev``。
    """
    height, width = _check_level_shape(prev)
    if height == 0 or width == 0:
        return []
    _check_depths(prev, "build_level 的输入")

    result: list[list[float]] = []
    for y in range(0, height, 2):
        row: list[float] = []
        y_end = min(y + 2, height)
        for x in range(0, width, 2):
            x_end = min(x + 2, width)
            block_max = prev[y][x]
            for by in range(y, y_end):
                for bx in range(x, x_end):
                    value = prev[by][bx]
                    if value > block_max:
                        block_max = value
            row.append(block_max)
        result.append(row)
    return result


def build_mips(depth: list[list[float]], levels: int) -> list:
    """从原图逐层构建深度金字塔。

    ``levels`` 是层数上限，必须是正整数，否则抛出 :class:`HizError`。
    返回列表的第 0 项是原图，之后每一项由前一项调用 :func:`build_level`
    得到；尺寸至少一维严格变小且总像素数严格变小。构建在 1x1 层停止，
    因此 ``levels`` 足够时最后一层是 1x1；单像素图只有一层，即使
    ``levels`` 更大也不会重复生成 1x1 层。``depth`` 为空时返回 ``[]``。
    输入深度必须是有限且位于 ``[0, 1]`` 的数值。
    """
    if not _is_int(levels) or levels <= 0:
        raise HizError("levels 必须是正整数")

    height, width = _check_level_shape(depth)
    if height == 0 or width == 0:
        return []
    _check_depths(depth, "build_mips 的输入")

    mips: list = [_copy_level(depth)]
    while len(mips) < levels:
        current = mips[-1]
        nxt = build_level(current)
        cur_h, cur_w = len(current), len(current[0])
        nxt_h = len(nxt)
        nxt_w = len(nxt[0]) if nxt_h else 0

        # 尺寸必须严格缩小：任一维不增大，且总像素数严格减少。
        if nxt_h > cur_h or nxt_w > cur_w:
            break
        if nxt_h * nxt_w >= cur_h * cur_w:
            break
        mips.append(nxt)
    return mips


# ---------------------------------------------------------------------------
# 选层与遮挡测试
# ---------------------------------------------------------------------------


def pick_level(rect: dict, levels: int) -> int:
    """选择仍能覆盖矩形的最小 0-based 层级索引。

    ``levels`` 是可用层数，必须是正整数。返回最小的非负整数 ``l``，
    使得第 ``l`` 层的一个像素跨度 ``2**l`` 不小于 ``max(rect["w"],
    rect["h"])``；若所需层级超过可用范围，则返回 ``levels - 1``。
    例如 3x2 的矩形返回 2（2**1 < 3 <= 2**2），不能向下取整到 1。
    ``rect`` 的宽高必须为正，非法时抛出 :class:`HizError`。
    """
    if not _is_int(levels) or levels <= 0:
        raise HizError("levels 必须是正整数")
    _, _, w, h = _check_positive_rect(rect)

    span = max(w, h)
    # 最小的 l 使 2**l >= span，即对 (span - 1) 取二进制长度。
    level = (span - 1).bit_length()
    return min(level, levels - 1)


def occluded(rect: dict, level: int, mips: list, depth: float) -> bool:
    """保守判断矩形在指定层级上是否被完全遮挡。

    ``mips`` 是 :func:`build_mips` 的结果，``level`` 是其有效下标。
    只考察 ``mips[level]`` 中与半开矩形相交的像素，用第 0 层坐标右移
    ``level`` 位映射；右边界/下边界之外以及图外的像素不能参与判断。
    取这些层级像素的最大深度 ``block_max``，当满足
    ``depth + EPS >= block_max``（即查询对象不近于块内任何已绘几何，
    含贴面与容差）时返回 ``True``，否则返回 ``False``。矩形与图无交集
    时返回 ``False``。``depth`` 必须是 ``[0, 1]`` 内的有限数；``level``、
    ``mips`` 或矩形非法时抛出 :class:`HizError`。
    """
    if not _is_valid_depth(depth):
        raise HizError("occluded 的 depth 必须是 [0, 1] 内的有限数")
    if not _is_int(level) or level < 0:
        raise HizError("level 必须是非负整数")
    if not isinstance(mips, (list, tuple)):
        raise HizError("mips 必须是深度层列表")
    if level >= len(mips):
        raise HizError("level 超出 mips 的可用范围")

    grid = mips[level]
    grid_h, grid_w = _check_level_shape(grid)
    if grid_h == 0 or grid_w == 0:
        raise HizError("mips[level] 不能是空层")

    x, y, w, h = _check_positive_rect(rect)
    right = x + w
    bottom = y + h

    # 与图（第 0 层坐标）无交集。
    if right <= 0 or bottom <= 0:
        return False

    # 与半开矩形相交的层级像素下标范围，右/下边界之外不读取。
    x_start = max(0, x >> level)
    x_stop = min(grid_w, ((right - 1) >> level) + 1)
    y_start = max(0, y >> level)
    y_stop = min(grid_h, ((bottom - 1) >> level) + 1)
    if x_start >= x_stop or y_start >= y_stop:
        return False

    block_max = None
    for gy in range(y_start, y_stop):
        row = grid[gy]
        for gx in range(x_start, x_stop):
            value = row[gx]
            if block_max is None or value > block_max:
                block_max = value

    assert block_max is not None
    return depth + EPS >= block_max


# ---------------------------------------------------------------------------
# 矩形工具
# ---------------------------------------------------------------------------


def clamp_rect(rect: dict, w: int, h: int) -> dict:
    """把矩形裁剪到 ``[0, w) x [0, h)`` 的图像范围内。

    返回新的 ``{"x", "y", "w", "h"}`` 字典，绝不产生负坐标或越界读取。
    部分越界时保留交集；完全在图像左侧/上方时坐标钳到 0，完全在右侧/
    下方时坐标钳到 ``w``/``h``，此时宽或高为 0。``w``、``h`` 必须是非负
    整数；矩形 ``x``、``y`` 可为任意整数，但宽高必须是非负整数，否则抛出
    :class:`HizError`。
    """
    if not _is_int(w) or not _is_int(h) or w < 0 or h < 0:
        raise HizError("图像宽高必须是非负整数")

    x, y, rect_w, rect_h = _check_rect_fields(rect)

    new_x = min(max(x, 0), w)
    new_y = min(max(y, 0), h)
    # 与 [0, w) 的交集宽度；交集为空时自然得到 0。
    new_w = max(0, min(x + rect_w, w) - new_x)
    new_h = max(0, min(y + rect_h, h) - new_y)
    return {"x": new_x, "y": new_y, "w": new_w, "h": new_h}


def _rectangles_overlap(a: tuple[int, int, int, int],
                        b: tuple[int, int, int, int]) -> bool:
    """两个半开矩形是否有面积为正的交集（仅边/角相接不算）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax < bx + bw
        and bx < ax + aw
        and ay < by + bh
        and by < ay + ah
    )


def merge_rects(rects: list) -> list:
    """合并相互重叠的查询矩形，减少深度回读次数。

    两个矩形只有在交集面积为正时才算重叠；仅边或角相接不合并。重叠关系
    具有传递性，合并后的矩形是参与合并的所有矩形的轴对齐包围盒。结果按
    ``(y, x, w, h)`` 升序排列，输入列表和字典不被修改。空列表返回 ``[]``。
    ``rects`` 中每个矩形都必须合法，否则抛出 :class:`HizError`。
    """
    if not isinstance(rects, (list, tuple)):
        raise HizError("rects 必须是矩形列表")

    boxes = [_check_positive_rect(rect) for rect in rects]
    count = len(boxes)
    if count == 0:
        return []

    # 并查集求重叠关系的连通分量（重叠具有传递性）。
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(count):
        for j in range(i + 1, count):
            if _rectangles_overlap(boxes[i], boxes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    merged: list[dict] = []
    for members in groups.values():
        xs = [boxes[i][0] for i in members]
        ys = [boxes[i][1] for i in members]
        rights = [boxes[i][0] + boxes[i][2] for i in members]
        bottoms = [boxes[i][1] + boxes[i][3] for i in members]
        min_x, min_y = min(xs), min(ys)
        merged.append(
            {
                "x": min_x,
                "y": min_y,
                "w": max(rights) - min_x,
                "h": max(bottoms) - min_y,
            }
        )

    merged.sort(key=lambda r: (r["y"], r["x"], r["w"], r["h"]))
    return merged


# ---------------------------------------------------------------------------
# 校验与批量入口
# ---------------------------------------------------------------------------


def validate_mips(mips: list) -> list[str]:
    """检查金字塔并返回问题标签列表，合法时返回 ``[]``。

    检查三类问题：

    * ``"empty_level"``：某层没有行，或某行没有像素；
    * ``"size_not_decreasing"``：后一层任一维大于前一层，或总像素数没有
      严格减少（因此合法的非方形链如 8x1 -> 4x1 -> 2x1 -> 1x1 可以接受）；
    * ``"depth_out_of_range"``：深度不是数值、不是有限数，或不在
      ``[0.0, 1.0]`` 内；每个违规像素各记一条。

    每个问题是一个字符串，按层号和层内检查顺序追加。函数不抛异常来表示
    这些数据问题；``mips`` 为空返回 ``[]``。NaN 与无穷大必须报告为
    ``"depth_out_of_range"``，不能静默传播。
    """
    issues: list[str] = []
    if not isinstance(mips, (list, tuple)) or len(mips) == 0:
        return issues

    prev_size: tuple[int, int] | None = None

    for level in mips:
        if not isinstance(level, (list, tuple)):
            issues.append("empty_level")
            prev_size = None
            continue

        if len(level) == 0:
            issues.append("empty_level")
            prev_size = None
            continue

        first = level[0]
        if not isinstance(first, (list, tuple)) or len(first) == 0:
            issues.append("empty_level")
            prev_size = None
            continue

        height = len(level)
        width = len(first)

        has_empty_row = False
        for row in level:
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                has_empty_row = True
                break
        if has_empty_row:
            issues.append("empty_level")
            prev_size = None
            continue

        if prev_size is not None:
            prev_h, prev_w = prev_size
            if (
                height > prev_h
                or width > prev_w
                or height * width >= prev_h * prev_w
            ):
                issues.append("size_not_decreasing")
        prev_size = (height, width)

        for row in level:
            if not isinstance(row, (list, tuple)):
                issues.append("empty_level")
                continue
            for value in row:
                if not _is_valid_depth(value):
                    issues.append("depth_out_of_range")

    return issues


def hiz_batch(frames: list, config: dict) -> dict:
    """批量构建多个帧的 Hi-Z 金字塔，并保证全有或全无。

    ``config`` 必须含正整数 ``"levels"``（层数上限），否则抛出
    :class:`HizError`。每个帧都必须是非空矩形深度图，所有深度有限且位于
    ``[0, 1]``；任一帧非法时先整体失败并抛出 :class:`HizError`，不返回
    部分结果，也不修改 ``frames`` 或 ``config``。各帧必须拥有彼此独立的
    mip 链，不能共享同一个可变列表。

    成功时返回 ``{"mips": [第 0 帧的链, 第 1 帧的链, ...],
    "count": 帧数, "levels": 层数上限}``，链的顺序与 ``frames`` 一致。
    ``frames`` 为空时返回 ``{"mips": [], "count": 0, "levels": levels}``。
    """
    if not isinstance(config, dict) or "levels" not in config:
        raise HizError("config 必须包含 levels")
    levels = config["levels"]
    if not _is_int(levels) or levels <= 0:
        raise HizError("config['levels'] 必须是正整数")

    if not isinstance(frames, (list, tuple)):
        raise HizError("frames 必须是帧列表")

    # 先全量校验：任一帧非法就整体失败，避免产生部分结果。
    for index, frame in enumerate(frames):
        height, width = _check_level_shape(frame)
        if height == 0 or width == 0:
            raise HizError(f"第 {index} 帧不能为空")
        _check_depths(frame, f"hiz_batch 的第 {index} 帧")

    # 每帧单独构建，build_mips 会复制第 0 层，因此链之间互不共享。
    chains = [build_mips(frame, levels) for frame in frames]
    return {"mips": chains, "count": len(chains), "levels": levels}
