"""在 Windows 上用桌面窗口打开本地记账。

双击桌面图标时会执行这个文件。它只连接本机页面，不把账本发到网上。
如果 8501 端口还没有服务，就在后台启动 Streamlit；关掉窗口时，只结束这次启动的服务。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PORT = 8501
URL = f"http://127.0.0.1:{PORT}"
CREATE_NO_WINDOW = 0x08000000


def port_is_open(port: int) -> bool:
    """检查本机端口是否已经有记账服务。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def python_executable() -> Path:
    """优先使用项目虚拟环境，避免点到系统里别的 Python。"""
    scripts = ROOT / ".venv" / "Scripts"
    for name in ("pythonw.exe", "python.exe"):
        candidate = scripts / name
        if candidate.exists() and candidate != Path(sys.executable):
            return scripts / "python.exe"
    return Path(sys.executable)


def start_server() -> subprocess.Popen[bytes]:
    """在后台启动本机页面，不弹出黑色命令行窗口。"""
    command = [
        str(python_executable()),
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        creationflags=CREATE_NO_WINDOW,
    )


def wait_until_ready(process: subprocess.Popen[bytes], timeout: float = 40) -> None:
    """等待页面开始接受连接。启动失败时抛出明确错误。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_open(PORT):
            return
        if process.poll() is not None:
            raise RuntimeError("记账服务启动后立即退出。请在项目目录运行 streamlit run app.py 查看原因。")
        time.sleep(0.25)
    raise RuntimeError("记账页面在 40 秒内没有打开。")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """结束本次启动的服务及其子进程。"""
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )


def tell_user(message: str) -> None:
    """没有命令行窗口时，用系统对话框说明问题。"""
    if sys.platform != "win32":
        print(message)
        return
    import ctypes

    ctypes.windll.user32.MessageBoxW(0, message, "本地智能记账", 0x40)


def open_window() -> bool:
    """打开桌面窗口。返回 True 表示用户已关闭窗口，可以停止本次启动的服务。"""
    try:
        import webview
    except ImportError:
        tell_user("未安装窗口组件。页面会改用浏览器打开，服务会继续留在本机。")
        webbrowser.open(URL)
        return False
    try:
        webview.create_window("本地智能记账", URL, width=1280, height=860, min_size=(960, 680))
        webview.start()
    except Exception:
        tell_user("没能创建桌面窗口，已改用浏览器打开同一页面。")
        webbrowser.open(URL)
        return False
    return True


def main() -> None:
    """已有服务时只开窗口；否则先启动服务，关窗口后再关掉服务。"""
    server = None
    close_server = False
    try:
        if not port_is_open(PORT):
            server = start_server()
            wait_until_ready(server)
        close_server = open_window()
    except Exception as exc:
        tell_user(str(exc))
    finally:
        if server is not None and close_server:
            stop_server(server)


if __name__ == "__main__":
    main()
