"""Loopback-only HTTP interface; no shell commands or user-chosen output paths."""

import argparse
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
from urllib.parse import parse_qs, unquote, urlsplit

from sno_mdp.config import Config
from .lab import ALLOWED_ARTIFACTS, Lab, MAX_CASES, ROOT, plan_request


STATIC = Path(__file__).parent / "static"


def make_server(root=ROOT, port=8765, start_worker=True):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        server_version = "SNO-MDP-Lab/1.0"

        def log_message(self, format, *args):
            if len(args) < 2 or str(args[1]) not in ("200", "202", "304"):
                super().log_message(format, *args)

        def send(self, data, mime="application/json", status=200, compress=True):
            if mime == "application/json" and not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            elif isinstance(data, str):
                data = data.encode()
            zipped = (
                compress
                and len(data) > 1024
                and "gzip" in self.headers.get("Accept-Encoding", "")
            )
            if zipped:
                data = gzip.compress(data)
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            if zipped:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.end_headers()
            self.wfile.write(data)

        def check_local(self, mutation=False):
            hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in hosts or (origin and origin != "http://" + host):
                raise PermissionError("Only same-origin local requests are allowed.")
            if mutation and not secrets.compare_digest(
                self.headers.get("X-Lab-Token", ""), token
            ):
                raise PermissionError(
                    "Missing local session token. Reload the dashboard."
                )

        def do_GET(self):
            try:
                self.check_local()
                url = urlsplit(self.path)
                path = unquote(url.path)
                query = parse_qs(url.query)
                if path == "/api/status":
                    return self.send(
                        {
                            "frozen": lab.frozen_status(),
                            "token": token,
                            "defaults": Config.load(
                                lab.root / "configs/demo_source.json"
                            ).as_dict()
                            if (lab.root / "configs/demo_source.json").exists()
                            else Config().as_dict(),
                            "max_cases": MAX_CASES,
                            "output": "results/interactive/",
                        }
                    )
                if path == "/api/catalog":
                    if query.get("refresh") == ["1"]:
                        lab.refresh()
                    return self.send(lab.catalog())
                if path == "/api/jobs":
                    return self.send(lab.jobs_list())
                if path.startswith("/api/run/"):
                    return self.send(lab.payload(path.split("/")[-1]))
                if path.startswith("/api/series/"):
                    return self.send(lab.series(path.split("/")[-1]))
                if path == "/api/inspect":
                    return self.send(
                        lab.inspect(
                            query["run"][0],
                            int(query["step"][0]),
                            int(query["x"][0]),
                            int(query["y"][0]),
                        )
                    )
                if path.startswith("/artifact/"):
                    _, _, identifier, filename = path.split("/", 3)
                    if filename not in ALLOWED_ARTIFACTS:
                        raise ValueError("Artifact is not on the download list.")
                    directory, _ = lab.resolve_run(identifier)
                    artifact = lab._inside_results(directory / filename)
                    artifact.relative_to(directory)
                    mime = (
                        mimetypes.guess_type(filename)[0] or "application/octet-stream"
                    )
                    return self.send(
                        artifact.read_bytes(),
                        mime,
                        compress=not filename.endswith(".gz"),
                    )
                name = "index.html" if path == "/" else path[1:]
                if name not in ("index.html", "app.js", "style.css"):
                    return self.send({"error": "Not found"}, status=404)
                return self.send(
                    (STATIC / name).read_bytes(), mimetypes.guess_type(name)[0]
                )
            except (BrokenPipeError, ConnectionResetError):
                pass
            except PermissionError as exc:
                self.send({"error": str(exc)}, status=403)
            except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
                self.send({"error": str(exc)}, status=400)

        def do_POST(self):
            try:
                self.check_local(mutation=True)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 131072:
                    raise ValueError("Request must be valid JSON smaller than 128 KiB.")
                request = json.loads(self.rfile.read(length))
                path = urlsplit(self.path).path
                if path == "/api/plan":
                    cases = plan_request(request)
                    return self.send({"count": len(cases), "cases": cases})
                if path == "/api/compare":
                    return self.send(lab.compare(request.get("ids")))
                if path == "/api/jobs":
                    return self.send(lab.submit(request), status=202)
                if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    return self.send(lab.cancel(path.split("/")[3]))
                self.send({"error": "Not found"}, status=404)
            except PermissionError as exc:
                self.send({"error": str(exc)}, status=403)
            except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
                self.send({"error": str(exc)}, status=400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        lab = Lab(root, start_worker=start_worker)
    except Exception:
        server.server_close()
        raise
    server.daemon_threads = True
    server.lab = lab
    server.token = token
    return server


def main():
    parser = argparse.ArgumentParser(
        description="Local interactive SNO-MDP experiment lab"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        server = make_server(port=args.port)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(f"SNO-MDP Experiment Lab: http://127.0.0.1:{server.server_port}", flush=True)
    print("New experiments: results/interactive/ | Stop: Ctrl+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
