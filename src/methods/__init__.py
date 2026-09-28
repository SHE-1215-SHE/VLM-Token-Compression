"""压缩方法注册与构造。"""
from __future__ import annotations

from .fastv import FastV
from .visionzip import VisionZip
from .tome import ToMeVLM

METHOD_REGISTRY = {"fastv": FastV, "visionzip": VisionZip, "tome": ToMeVLM}


def build_method(name: str, budget: float, **kwargs):
    """按名字构造压缩方法实例。baseline 不走这里。"""
    if name not in METHOD_REGISTRY:
        raise KeyError(f"未知方法 {name}，可选: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name](budget=budget, **kwargs)
