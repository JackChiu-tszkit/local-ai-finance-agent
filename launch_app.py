"""Open the local ledger in a desktop window on Windows.

A desktop shortcut runs this file. It only connects to the page on this
computer and does not send the ledger anywhere else. If port 8501 is free,
Streamlit starts in the background. Closing the window stops only that process.
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
APP_TITLE = "Local ledger"


def port_is_open(port: int) -> bool:
    """Whether a ledger server is already listening on this computer."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def python_executable() -> Path:
    """Prefer the project virtual environment over another Python on PATH."""
    scripts = ROOT / ".venv" / "Scripts"
    for name in ("pythonw.exe", "python.exe"):
        candidate = scripts / name
        if candidate.exists() and candidate != Path(sys.executable):
            return scripts / "python.exe"
    return Path(sys.executable)


def start_server() -> subprocess.Popen[bytes]:
    """Start the local page in the background, without a console window."""
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
    """Wait until the page accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_is_open(PORT):
            return
        if process.poll() is not None:
            raise RuntimeError("The ledger server exited immediately. Run streamlit run app.py in the project folder to see why.")
        time.sleep(0.25)
    raise RuntimeError("The ledger page did not open within 40 seconds.")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """Stop the server started by this launch, including its child processes."""
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )


def tell_user(message: str) -> None:
    """Show a system dialog when there is no console window."""
    if sys.platform != "win32":
        print(message)
        return
    import ctypes

    ctypes.windll.user32.MessageBoxW(0, message, APP_TITLE, 0x40)


def open_window() -> bool:
    """Open the desktop window. True means the user closed it and the server may stop."""
    try:
        import webview
    except ImportError:
        tell_user("The window component is not installed. The page will open in a browser, and the server will keep running on this computer.")
        webbrowser.open(URL)
        return False
    try:
        webview.create_window(APP_TITLE, URL, width=1280, height=860, min_size=(960, 680))
        webview.start()
    except Exception:
        tell_user("Could not create a desktop window. The same page was opened in a browser.")
        webbrowser.open(URL)
        return False
    return True


def main() -> None:
    """Open a window when a server is already running. Otherwise start one and stop it when the window closes."""
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
