"""Talking to the AI texturing server.

Pure Python standard library, no bpy: everything here runs on a background
thread, and Blender's API must never be touched from another thread.

The request/response contract is described in README.md. In short:

    POST <server_url>   multipart/form-data
        image      cubemap.png (horizontal cross, transparent background)
        prompt     the user's text
        layout     "horizontal_cross"
        face_size  pixels per face
        faces      JSON: where each side sits, e.g. {"front": [1, 1], ...}

    The server answers with ONE of:
        - the image itself (Content-Type: image/png or image/jpeg)
        - JSON {"image": "<base64>"}  or  {"image_url": "https://..."}
        - JSON {"job_id": "...", "status_url": "..."} to be polled with GET
          until it returns one of the above, or {"status": "failed", "error": "..."}
"""

import base64
import json
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

USER_AGENT = "Blender-AI-Texture-Projector/1.0"
PENDING_STATES = {"queued", "pending", "running", "processing", "in_progress", "started"}
FAILED_STATES = {"failed", "error", "cancelled", "canceled"}


class ServerError(Exception):
    """Something went wrong; the message is shown to the user as-is."""


class Cancelled(Exception):
    pass


def is_local(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def check_url(url):
    """Return an error message for an unusable URL, or None."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return f"'{url}' isn't a valid http:// or https:// address"
    return None


def _ssl_context():
    # Blender's bundled Python may not see the system's certificates;
    # certifi (shipped with Blender) is the reliable fallback.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def encode_multipart(fields, files):
    """fields: {name: str}, files: {name: (filename, bytes, mime)}."""
    boundary = f"----blender-{uuid.uuid4().hex}"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            .encode() + str(value).encode("utf-8") + b"\r\n")
    for name, (filename, data, mime) in files.items():
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
             f'filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n').encode()
            + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _same_origin(a, b):
    pa, pb = urllib.parse.urlparse(a), urllib.parse.urlparse(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


def _looks_like_image(data):
    return data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff" or \
        (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


def _error_detail(body, content_type):
    if "json" in content_type:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                for key in ("error", "detail", "message"):
                    if payload.get(key):
                        return str(payload[key])
        except ValueError:
            pass
    text = body[:300].decode("utf-8", "replace").strip()
    return text or "no details"


class Client:
    def __init__(self, server_url, api_key="", timeout=300.0, poll_interval=2.0,
                 on_status=None, is_cancelled=None):
        self.server_url = server_url
        self.api_key = api_key
        self.deadline = time.monotonic() + timeout
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.on_status = on_status or (lambda message, progress=None: None)
        self.is_cancelled = is_cancelled or (lambda: False)
        self.ssl = _ssl_context()

    # -- low level -----------------------------------------------------------

    def _remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise ServerError(f"No result from the server within {self.timeout:.0f} s. "
                              "Increase the timeout in the add-on preferences")
        return left

    def _send(self, url, data=None, headers=None):
        """Make one request. Returns (status, content_type, body)."""
        headers = dict(headers or {})
        headers["User-Agent"] = USER_AGENT
        # Only send the key to the server itself, never to e.g. a storage URL
        # it points us to (pre-signed S3 links reject extra auth headers).
        if self.api_key and _same_origin(url, self.server_url):
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=self._remaining(),
                                        context=self.ssl) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as err:
            body = err.read() if err.fp else b""
            return err.code, err.headers.get("Content-Type", "") if err.headers else "", body
        except (socket.timeout, TimeoutError):
            raise ServerError(f"The server didn't answer within {self.timeout:.0f} s")
        except urllib.error.URLError as err:
            reason = getattr(err, "reason", err)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise ServerError(f"The server didn't answer within {self.timeout:.0f} s")
            raise ServerError(f"Couldn't reach the server: {reason}")
        except OSError as err:
            raise ServerError(f"Connection problem: {err}")

    def _check_cancel(self):
        if self.is_cancelled():
            raise Cancelled()

    # -- response handling ---------------------------------------------------

    def _handle(self, url, status, content_type, body, polling=False):
        """Turn a response into ("image", bytes) or ("poll", url, progress, msg)."""
        content_type = content_type.lower()
        if status >= 400:
            raise ServerError(f"Server error {status}: {_error_detail(body, content_type)}")

        if content_type.startswith("image/") or _looks_like_image(body):
            if not body:
                raise ServerError("The server sent an empty image")
            return ("image", body)

        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            raise ServerError("The server's reply was neither an image nor JSON: "
                              + _error_detail(body, ""))
        if not isinstance(payload, dict):
            raise ServerError("The server's JSON reply wasn't an object")

        state = str(payload.get("status", "")).lower()
        if state in FAILED_STATES:
            raise ServerError(f"The server reported a failure: "
                              f"{payload.get('error') or payload.get('message') or state}")

        if payload.get("image"):
            data = str(payload["image"])
            if data.startswith("data:"):           # data:image/png;base64,....
                data = data.split(",", 1)[-1]
            try:
                return ("image", base64.b64decode(data))
            except ValueError:
                raise ServerError("The 'image' field isn't valid base64")

        if payload.get("image_url"):
            image_url = urllib.parse.urljoin(url, payload["image_url"])
            self.on_status("Downloading result…")
            st, ct, data = self._send(image_url)
            if st >= 400:
                raise ServerError(f"Downloading the result failed ({st}): "
                                  f"{_error_detail(data, ct.lower())}")
            return ("image", data)

        poll_url = payload.get("status_url")
        if poll_url:
            poll_url = urllib.parse.urljoin(url, poll_url)
        elif payload.get("job_id"):
            poll_url = f"{self.server_url.rstrip('/')}/{payload['job_id']}"
        elif polling and (state in PENDING_STATES or status == 202):
            poll_url = url                         # keep polling the same address
        else:
            raise ServerError("The server's reply had no 'image', 'image_url' or "
                              "'status_url'. See README.md for the expected format")

        progress = payload.get("progress")
        try:
            progress = None if progress is None else max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            progress = None
        return ("poll", poll_url, progress, payload.get("message") or state or "")

    # -- the whole conversation ------------------------------------------------

    def request_texture(self, image_png, prompt, face_size, faces, layout="horizontal_cross",
                        extra_fields=None):
        fields = {"prompt": prompt, "layout": layout, "face_size": str(face_size),
                  "faces": json.dumps(faces)}
        fields.update(extra_fields or {})
        body, content_type = encode_multipart(
            fields, {"image": ("cubemap.png", image_png, "image/png")})

        self.on_status("Uploading…")
        result = self._handle(self.server_url,
                              *self._send(self.server_url, body, {"Content-Type": content_type}))
        while result[0] == "poll":
            _kind, poll_url, progress, message = result
            text = "Server is working"
            if message:
                text += f": {message}"
            self.on_status(text, progress)
            # Sleep in short steps so Cancel feels immediate.
            wake = time.monotonic() + self.poll_interval
            while time.monotonic() < wake:
                self._check_cancel()
                self._remaining()
                time.sleep(0.1)
            result = self._handle(poll_url, *self._send(poll_url), polling=True)
        self._check_cancel()
        return result[1]
