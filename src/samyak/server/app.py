"""Stdlib HTTP server for the local run-history viewer."""

from __future__ import annotations

import webbrowser
from contextlib import suppress
from http.server import ThreadingHTTPServer

from samyak.model.store import FileCatalogStore
from samyak.server.routes import ViewerHandler
from samyak.store.filesystem import RunStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 15500
_ALLOWED_HOSTS = frozenset({DEFAULT_HOST})


class ViewerBindError(OSError):
    """Raised when the local viewer cannot bind to the requested address."""


class ViewerServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True


def create_server(
    store: RunStore,
    catalog_store: FileCatalogStore | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> ThreadingHTTPServer:
    """Create a loopback HTTP server. Does not start serving."""
    if host not in _ALLOWED_HOSTS:
        raise ViewerBindError(f"local viewer only binds to {DEFAULT_HOST}, not {host}")
    handler = ViewerHandler.with_stores(store, catalog_store)
    try:
        return ViewerServer((host, port), handler)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise ViewerBindError(f"could not bind to {host}:{port} ({detail})") from exc


def viewer_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}/"


def serve_viewer(
    store: RunStore,
    catalog_store: FileCatalogStore | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> None:
    """Bind, optionally open a browser, and serve until interrupted."""
    server = create_server(store, catalog_store, host=host, port=port)
    url = viewer_url(server)
    print("Samyak local viewer", flush=True)
    print(url, flush=True)
    print("Local Samyak workspace — data stored on this machine.", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        with suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
