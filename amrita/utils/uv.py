"""异步 uv 执行层。

WebUI 的插件安装/卸载都要跑 ``uv``，而 ``uv add`` 会联网、可能耗时几十秒。
这里用 :func:`asyncio.create_subprocess_exec` 起子进程，逐行回吐输出，
既不会阻塞事件循环，也能把进度实时推给前端。

与 ``amctl.uv_util.UvOperator`` 的关系是**并存而非替代**：那个是同步实现，
服务于 CLI（``amctl create`` 末尾的 ``uv sync``）；这里是异步流式实现，
服务于 WebUI。两者的调用场景不同，硬合并会两头不讨好。

.. note::
   本模块**不在导入期**检查 uv 是否存在。``amctl.uv_util`` 在模块顶层就
   ``os.popen("uv --version")``，uv 不在 PATH 时直接抛异常——若被 WebUI
   顶层导入会让整个插件加载失败。这里改用惰性检查（:func:`uv_available`）。
"""

from __future__ import annotations

import asyncio
import shutil
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FileSnapshot",
    "UvCommandError",
    "UvNotAvailableError",
    "UvResult",
    "UvTimeoutError",
    "install_lock",
    "install_package",
    "remove_package",
    "run_uv",
    "summarize_uv_output",
    "uv_add",
    "uv_available",
    "uv_remove",
    "uv_sync",
]

LineCallback = Callable[[str], Awaitable[None]]

#: 单个任务保留的输出行数上限，防止异常输出撑爆内存
_MAX_LINES = 2000

_UV_BINARY = "uv"


class UvNotAvailableError(RuntimeError):
    """找不到 ``uv`` 可执行文件。"""


class UvCommandError(RuntimeError):
    """``uv`` 以非零退出码结束。"""

    def __init__(self, command: Sequence[str], returncode: int, output: str) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.output = output
        super().__init__(
            f"uv {' '.join(command)} 执行失败（退出码 {returncode}）:\n{output}"
        )


class UvTimeoutError(UvCommandError):
    """``uv`` 执行超时，已被强制终止。"""

    def __init__(self, command: Sequence[str], output: str) -> None:
        RuntimeError.__init__(
            self, f"uv {' '.join(command)} 执行超时，已终止:\n{output}"
        )
        self.command = tuple(command)
        self.returncode = -1
        self.output = output


@dataclass
class UvResult:
    """一次成功的 uv 调用结果。"""

    returncode: int
    output: str
    truncated: bool = False
    """输出行数超过 :data:`_MAX_LINES` 时置位，此时 ``output`` 只保留末尾部分。"""


def uv_available() -> bool:
    """``uv`` 是否在 PATH 中。惰性检查，不在导入期执行。"""
    return shutil.which(_UV_BINARY) is not None


async def run_uv(
    *args: str,
    cwd: Path | str | None = None,
    timeout: float = 180.0,
    on_line: LineCallback | None = None,
) -> UvResult:
    """异步执行 ``uv <args...>``。

    Args:
        args: 传给 uv 的参数，逐个传入，绝不经过 shell。
        cwd: 工作目录，通常是项目根。
        timeout: 超时秒数，超时后强杀子进程并抛 :class:`UvTimeoutError`。
        on_line: 每读到一行输出就 await 一次，用于向前端推流。

    Returns:
        退出码为 0 时的 :class:`UvResult`。

    Raises:
        UvNotAvailableError: 找不到 uv。
        UvTimeoutError: 超时。
        UvCommandError: 退出码非零。
    """
    if not uv_available():
        raise UvNotAvailableError(
            f"未找到 {_UV_BINARY} 可执行文件，请先安装 uv 并确保它在 PATH 中"
        )

    process = await asyncio.create_subprocess_exec(
        _UV_BINARY,
        *args,
        cwd=str(cwd) if cwd is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    lines: deque[str] = deque(maxlen=_MAX_LINES)
    state = {"truncated": False}

    async def _pump() -> None:
        assert process.stdout is not None
        async for raw in process.stdout:
            if len(lines) == _MAX_LINES:
                state["truncated"] = True
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            lines.append(line)
            if on_line is not None:
                await on_line(line)

    try:
        await asyncio.wait_for(_pump(), timeout=timeout)
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        raise UvTimeoutError(args, "\n".join(lines)) from None

    returncode = await process.wait()
    output = "\n".join(lines)
    if returncode != 0:
        raise UvCommandError(args, returncode, output)
    return UvResult(returncode=returncode, output=output, truncated=state["truncated"])


async def uv_add(package: str, *, cwd: Path | str | None = None, **kwargs) -> UvResult:
    """``uv add <package>``。"""
    return await run_uv("add", package, cwd=cwd, **kwargs)


async def uv_remove(
    package: str, *, cwd: Path | str | None = None, **kwargs
) -> UvResult:
    """``uv remove <package>``。"""
    return await run_uv("remove", package, cwd=cwd, **kwargs)


async def uv_sync(*, cwd: Path | str | None = None, **kwargs) -> UvResult:
    """``uv sync``。"""
    return await run_uv("sync", cwd=cwd, **kwargs)


_INSTALL_LOCK = asyncio.Lock()


def install_lock() -> asyncio.Lock:
    """全局安装锁：同一时刻只允许一个安装/卸载任务在跑。

    两个任务同时写 ``pyproject.toml`` 会互相覆盖，且 uv 自己也会争抢锁文件。
    """
    return _INSTALL_LOCK


@dataclass
class FileSnapshot:
    """一组文件的字节快照，用于失败回滚。"""

    files: dict[Path, bytes | None] = field(default_factory=dict)
    """路径 -> 原内容；原文件不存在时为 ``None``。"""

    @classmethod
    def capture(cls, paths: Iterable[Path | str]) -> FileSnapshot:
        """读取并保存这些文件的当前内容。"""
        files: dict[Path, bytes | None] = {}
        for raw in paths:
            path = Path(raw)
            files[path] = path.read_bytes() if path.is_file() else None
        return cls(files=files)

    def restore(self) -> None:
        """写回快照内容；原本不存在的文件会被删除。"""
        for path, data in self.files.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(data)


def project_lockfiles(root: Path | str) -> tuple[Path, Path]:
    """项目里需要跟着 uv 一起回滚的两个文件。"""
    base = Path(root)
    return base / "pyproject.toml", base / "uv.lock"


async def install_package(
    package: str,
    *,
    cwd: Path | str,
    timeout: float = 180.0,
    on_line: LineCallback | None = None,
) -> UvResult:
    """带锁与回滚的 ``uv add``。

    执行前快照 ``pyproject.toml`` 与 ``uv.lock``；uv 失败或超时时把它们还原，
    避免留下半个依赖声明。

    .. note::
       文件能还原，但已装进虚拟环境的包不会自动卸掉；真要彻底复位再跑一次
       ``uv sync`` 即可。
    """
    async with install_lock():
        snapshot = FileSnapshot.capture(project_lockfiles(cwd))
        try:
            return await uv_add(package, cwd=cwd, timeout=timeout, on_line=on_line)
        except UvCommandError:
            snapshot.restore()
            raise


async def remove_package(
    package: str,
    *,
    cwd: Path | str,
    timeout: float = 180.0,
    on_line: LineCallback | None = None,
) -> UvResult:
    """带锁与回滚的 ``uv remove``。"""
    async with install_lock():
        snapshot = FileSnapshot.capture(project_lockfiles(cwd))
        try:
            return await uv_remove(package, cwd=cwd, timeout=timeout, on_line=on_line)
        except UvCommandError:
            snapshot.restore()
            raise


_UV_ERROR_MARKERS = ("Because ", "No solution found", "error:")


def summarize_uv_output(output: str, *, limit: int = 320) -> str:
    """从 uv 的失败输出里提炼一句能看懂的话。

    uv 的解析失败会打一整段带框线、还带多行折行的说明，直接甩给用户没法读。
    这里先把框线和折行抹平成一整段，再从关键位置截出第一句。
    """
    lines: list[str] = []
    for raw in output.splitlines():
        cleaned = raw.strip().lstrip("│╰─▶×\t ").strip()
        if cleaned:
            lines.append(cleaned)
    text = " ".join(lines)
    if not text:
        return "uv 执行失败，详情见任务输出"

    # 优先取 "Because ..." 那句，它才说明冲突原因；其余多为噪声
    idx = text.find("Because ")
    if idx == -1:
        for marker in _UV_ERROR_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                break
    if idx > 0:
        text = text[idx:]

    if len(text) <= limit:
        return text
    cut = text.find(". ", 0, limit)
    if cut == -1:
        cut = limit
    else:
        cut += 1
    return text[:cut].strip() + " …"
