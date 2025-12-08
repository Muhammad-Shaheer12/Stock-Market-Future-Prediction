import time, threading, webbrowser
import uvicorn

URL = "http://127.0.0.1:8000/"


def open_browser():
    def _open():
        time.sleep(1.2)
        try:
            webbrowser.open(URL)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    open_browser()
    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=True)
