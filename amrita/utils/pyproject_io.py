"""``pyproject.toml`` 的定位与读写。

从 ambot-inlinectl 的插件管理逻辑中提取而来，供 CLI（ambot）与 WebUI 共用，
避免同一套「改 pyproject」的实现在两处各自漂移。

本模块刻意不依赖 ``click`` / ``nonebot``：调用方负责把
:class:`PyprojectNotFoundError` 转换成自己的错误类型。

.. note::
   ``[tool]`` 段一旦被非 tool 的表（例如 ``[dependency-groups]``）隔断，
   tomlkit 会把它包成 :class:`~tomlkit.container.OutOfOrderTableProxy` 而不是
   ``Table``。两者都必须认，否则会误判成「不存在」并整段覆盖，静默丢掉配置。
"""

from __future__ import annotations

import os
from importlib.metadata import packages_distributions
from pathlib import Path
from typing import Any, Literal

import tomlkit
from tomlkit import TOMLDocument
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import Array, Table

__all__ = [
    "PluginTarget",
    "PyprojectNotFoundError",
    "find_pyproject",
    "load_pyproject",
    "modify_plugin_list",
    "read_plugin_list",
    "resolve_module_name",
    "save_pyproject",
]

PluginTarget = Literal["amrita", "nonebot"]

# tomlkit 里「表」的两种形态：常规 Table，以及键被打断时的乱序代理
_TABLE_TYPES: tuple[type, ...] = (Table, OutOfOrderTableProxy)


class PyprojectNotFoundError(FileNotFoundError):
    """在起始目录及其所有祖先目录中都未找到 ``pyproject.toml``。"""


def find_pyproject(start_dir: Path | str | None = None) -> Path | None:
    """从 ``start_dir`` 逐级向上查找 ``pyproject.toml``。

    Args:
        start_dir: 查找起点。``None`` 表示当前工作目录；传入文件路径时从其父目录开始。

    Returns:
        找到时返回绝对路径；抵达文件系统根目录仍未命中时返回 ``None``。
    """
    current = Path.cwd() if start_dir is None else Path(start_dir).resolve()
    if current.is_file():
        current = current.parent
    while True:
        candidate = current / "pyproject.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def load_pyproject(path: Path | str | None = None) -> tuple[Path, TOMLDocument]:
    """定位并解析 ``pyproject.toml``。

    Args:
        path: 显式路径；``None`` 时交由 :func:`find_pyproject` 向上查找。

    Returns:
        ``(绝对路径, 解析结果)``。解析走 tomlkit，保留注释与原始排版。

    Raises:
        PyprojectNotFoundError: 未找到文件。
    """
    target = Path(path) if path is not None else find_pyproject()
    if target is None:
        raise PyprojectNotFoundError("未找到 pyproject.toml")
    target = target.resolve()
    if not target.is_file():
        raise PyprojectNotFoundError(f"未找到 pyproject.toml: {target}")
    return target, tomlkit.parse(target.read_text(encoding="utf-8"))


def save_pyproject(path: Path | str, doc: TOMLDocument) -> None:
    """原子写回 ``pyproject.toml``。

    先写同目录下的临时文件再 :func:`os.replace`，避免中途失败留下半个文件
    ——那会让 bot 直接起不来。
    """
    target = Path(path)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    os.replace(tmp, target)


def _require_table(item: Any, name: str) -> Any:
    """确认 ``item`` 是 tomlkit 的表，缺失返回 ``None``，类型不符直接报错。

    宁可在类型不符时报错，也不要覆盖——覆盖会静默丢掉整段配置。
    """
    if item is None:
        return None
    if isinstance(item, _TABLE_TYPES):
        return item
    raise TypeError(f"{name} 不是 TOML 表，拒绝写入")


def _get_plugins_array(doc: TOMLDocument, target: PluginTarget) -> Array | None:
    """只读地取出 ``[tool.<target>].plugins``，缺失或类型不符时返回 ``None``。"""
    tool = _require_table(doc.get("tool"), "[tool]")
    if tool is None:
        return None
    section = _require_table(tool.get(target), f"[tool.{target}]")
    if section is None:
        return None
    plugins = section.get("plugins")
    return plugins if isinstance(plugins, Array) else None


def _ensure_plugins_section(
    doc: TOMLDocument, target: PluginTarget
) -> tuple[Any, bool]:
    """取出（必要时创建）``[tool.<target>]`` 表。

    Returns:
        ``(表对象, 是否新建)``。新建时需要由调用方在末尾补一个空行。
    """
    tool = _require_table(doc.get("tool"), "[tool]")
    if tool is None:
        tool = tomlkit.table()
        doc["tool"] = tool
    section = _require_table(tool.get(target), f"[tool.{target}]")
    if section is not None:
        return section, False
    section = tomlkit.table()
    tool[target] = section
    return section, True


def read_plugin_list(
    target: PluginTarget = "amrita", path: Path | str | None = None
) -> list[str]:
    """读取 ``[tool.<target>].plugins`` 的条目。"""
    _, doc = load_pyproject(path)
    plugins = _get_plugins_array(doc, target)
    return [] if plugins is None else [str(item) for item in plugins]


def modify_plugin_list(
    package: str,
    *,
    target: PluginTarget = "amrita",
    remove: bool = False,
    path: Path | str | None = None,
) -> bool:
    """在 ``[tool.<target>].plugins`` 中增删一个条目。

    ``package`` 原样写入，不做归一化——调用方应传入模块名，可用
    :func:`resolve_module_name` 从发行包名换算。

    Returns:
        ``True`` 表示实际发生修改；``False`` 表示条目已存在（新增）或本就不存在（移除）。

    Raises:
        PyprojectNotFoundError: 未找到 ``pyproject.toml``。
    """
    pyproject, doc = load_pyproject(path)
    section, created = _ensure_plugins_section(doc, target)

    plugins = section.get("plugins")
    if not isinstance(plugins, Array):
        plugins = tomlkit.array()
        section["plugins"] = plugins
    plugins.multiline(True)

    entries = [str(item) for item in plugins]
    if remove:
        if package not in entries:
            return False
        # tomlkit 的 Array 没有按值删除的接口，只能重建一个
        remaining = tomlkit.array()
        remaining.multiline(True)
        for item in plugins:
            if str(item) != package:
                remaining.append(item)
        section["plugins"] = remaining
    else:
        if package in entries:
            return False
        plugins.append(package)

    if created:
        # 新插入的表会紧贴下一段，补一个空行维持可读性
        section.add(tomlkit.nl())

    save_pyproject(pyproject, doc)
    return True


def resolve_module_name(project_link: str) -> str:
    """把发行包名换算成可导入的模块名。

    优先用 :func:`importlib.metadata.packages_distributions` 反查——包已安装时最准；
    查不到（未安装、元数据缺失）时退回 ``-`` → ``_`` 的字符串替换。
    """
    normalized = project_link.replace("-", "_").lower()
    try:
        mapping = packages_distributions()
    except Exception:
        return normalized

    candidates = [
        module
        for module, dists in mapping.items()
        if any(dist.replace("-", "_").lower() == normalized for dist in dists)
    ]
    if normalized in candidates:
        return normalized
    if candidates:
        return sorted(candidates)[0]
    return normalized
