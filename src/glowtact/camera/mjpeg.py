"""Stream a GlowTact over HTTP as MJPEG -- view in any browser, no display.

``multipart/x-mixed-replace`` is the classic webcam-in-a-browser transport: the
server pushes a never-ending sequence of JPEG frames on one HTTP response and
the browser swaps each in place. No JavaScript, no extra deps -- just OpenCV's
JPEG encoder and the stdlib http.server.

    from glowtact.camera import GlowTactCamera, serve_mjpeg
    with GlowTactCamera() as cam:
        serve_mjpeg(cam, port=8000)          # open http://<host>:8000/

One camera fans out to many viewers: every client pulls from the same
drop-old reader, so a slow client just sees fewer frames.
"""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_INDEX = b"""<!doctype html><meta charset=utf-8>
<title>GlowTact stream</title>
<body style="margin:0;background:#111;display:flex;justify-content:center">
<img src="/stream.mjpg" style="max-width:100vw;max-height:100vh">
"""


def _make_handler(cam, quality, diff_mode, gain):
    import cv2

    def encode(frame):
        out = (cam.difference(frame, diff_mode, gain)
               if diff_mode else frame)
        ok, buf = cv2.imencode(".jpg", out,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass                              # quiet; callers can add their own

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_INDEX)))
                self.end_headers()
                self.wfile.write(_INDEX)
                return
            if self.path not in ("/stream.mjpg", "/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    item = cam._reader.wait_for_new(last, timeout=5.0)
                    if item is None:
                        continue
                    last, frame, _ = item
                    jpeg = encode(frame)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                          # client navigated away

    return Handler


def serve_mjpeg(cam, host="0.0.0.0", port=8000, *, quality=80,
                diff_mode=None, gain=1.0, background=False):
    """Serve ``cam`` as MJPEG at http://host:port/.

    Blocks until Ctrl-C unless ``background=True``, in which case the server
    runs on a daemon thread and the ThreadingHTTPServer is returned so the
    caller can ``.shutdown()`` it. ``diff_mode`` streams the difference image
    (needs ``cam.set_baseline()`` first).
    """
    handler = _make_handler(cam, quality, diff_mode, gain)
    httpd = ThreadingHTTPServer((host, port), handler)
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return httpd
