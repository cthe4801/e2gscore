"""harness.py -- the one way the paper-reproduction modules read data: by path inside this repository.

Every module names its inputs by their repository path (scores/..., inputs/..., data/..., paper/data/...).
E2G_RELEASE points at another copy of the repository; by default it is the repository this file sits in.
"""
from __future__ import annotations

import gzip
import io
import os

ROOT = os.environ.get("E2G_RELEASE") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(rel: str) -> str:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{rel}: not found under {ROOT}")
    return p


def open_text(rel: str):
    """Open a repository text file, gzipped or plain."""
    p = resolve(rel)
    if p.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8")
    return open(p, encoding="utf-8")


def open_binary(rel: str):
    return open(resolve(rel), "rb")
