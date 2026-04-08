# -*- coding: utf-8 -*-
"""数据集子集工具：限量采样越狱/良性，优先保留文言文/CC-BOS 风格样本。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

WENYAN_SOURCES = frozenset({"wenyan_cc_bos_style"})


def is_wenyan_item(item: Dict[str, Any]) -> bool:
    return item.get("source") in WENYAN_SOURCES


def subset_jailbreak(items: List[dict], limit: int, prefer_wenyan: bool) -> List[dict]:
    if limit <= 0 or not items:
        return list(items)
    if len(items) <= limit:
        return list(items)
    if not prefer_wenyan:
        return items[:limit]
    wy = [x for x in items if is_wenyan_item(x)]
    rest = [x for x in items if not is_wenyan_item(x)]
    out = wy[:limit]
    need = limit - len(out)
    if need > 0:
        out.extend(rest[:need])
    return out


def subset_benign(items: List[dict], limit: int) -> List[dict]:
    if limit <= 0 or not items:
        return list(items)
    if len(items) <= limit:
        return list(items)
    return items[:limit]


def apply_category_limits(
    split_data: Dict[str, Any],
    jailbreak_limit: int = 0,
    benign_limit: int = 0,
    prefer_wenyan: bool = False,
) -> Dict[str, Any]:
    out = deepcopy(split_data)
    if jailbreak_limit > 0 and "jailbreak" in out:
        out["jailbreak"] = subset_jailbreak(out["jailbreak"], jailbreak_limit, prefer_wenyan)
    if benign_limit > 0 and "benign" in out:
        out["benign"] = subset_benign(out["benign"], benign_limit)
    return out


def count_wenyan_jailbreak(items: List[dict]) -> int:
    return sum(1 for x in items if is_wenyan_item(x))
