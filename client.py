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
        - JSON {"image": "<base64>"}, {"image_url": "..."} or {"result_url": "..."}
        - JSON {"job_id": "...", "status": "queued", "status_url": "..."}: a
          background job. The client then polls status_url with GET. Each poll
          returns {"status": "queued", "position": 2} / {"status": "running",
          "progress": 0.4, "message": "..."} until it returns one of the results
          above, or {"status": "failed", "error": "..."}.

Progress is reported through on_status(info) with info["phase"] one of:
    uploading, queued, running, downloading, retrying
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

USER_AGENT = "Blender-AI-Texture-Projector/1.1"

QUEUED_STATES = {"queued", "pending", "waiting", "submitted"}
RUNNING_STATES = {"running", "processing", "in_progress", "started", "generating"}
DONE_STATES = {"done", "completed", "complete", "succeeded", "success", "finished"}
FAILED_STATES = {"failed", "error", "cancelled", "canceled"}
TRANSIENT_HTTP = {502, 503, 504}     # proxy/gateway hiccups while polling

POLL_REQUEST_TIMEOUT = 30            # seconds for one status check
DOWNLOAD_TIMEOUT = 120               # seconds to download the result image


class ServerError(Exception):
    """Something went wrong; the message is shown to the user as-is."""


class TransientError(ServerError):
    """No usable answer this time (connection dropped, gateway error).
    Worth retrying while polling, because the job lives on in the server."""


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


def _number(value, lo=None, hi=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


class Client:
    def __init__(self, server_url, api_key="", timeout=300.0, poll_interval=2.0,
                 on_status=None, is_cancelled=None, max_retries=6):
        self.server_url = server_url
        self.api_key = api_key
        self.timeout = timeout              # max time a job may RUN (queue time is free)
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.on_status = on_status or (lambda info: None)
        self.is_cancelled = is_cancelled or (lambda: False)
        self.ssl = _ssl_context()
        self.status_url = None              # remembered so a cancel can tell the server
        self.run_deadline = None            # set when the server starts the job

    # -- low level -----------------------------------------------------------

    def _send(self, url, data=None, headers=None, method=None, request_timeout=30.0):
        """Make one request. Returns (status, content_type, body).
        Raises TransientError if no answer arrived at all."""
        headers = dict(headers or {})
        headers["User-Agent"] = USER_AGENT
        # Only send the key to the server itself, never to e.g. a storage URL
        # it points us to (pre-signed S3 links reject extra auth headers).
        if self.api_key and _same_origin(url, self.server_url):
            headers["Authorization"] = f"Bearer {self.api_key}"
        method = method or ("POST" if data is not None else "GET")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=request_timeout, context=self.ssl) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as err:
            body = err.read() if err.fp else b""
            return err.code, err.headers.get("Content-Type", "") if err.headers else "", body
        except (socket.timeout, TimeoutError):
            raise TransientError(f"no answer within {request_timeout:.0f} s")
        except urllib.error.URLError as err:
            reason = getattr(err, "reason", err)
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise TransientError(f"no answer within {request_timeout:.0f} s")
            raise TransientError(f"couldn't reach the server ({reason})")
        except OSError as err:
            raise TransientError(f"connection problem ({err})")

    def _status(self, **info):
        self.on_status(info)

    def _sleep(self, seconds):
        """Wait in short steps so Cancel feels immediate."""
        wake = time.monotonic() + seconds
        while time.monotonic() < wake:
            if self.is_cancelled():
                raise Cancelled()
            if self.run_deadline is not None and time.monotonic() > self.run_deadline:
                raise ServerError(f"The job has been running on the server for more than "
                                  f"{self.timeout:.0f} s. Increase the timeout in the "
                                  "add-on preferences if that's normal")
            time.sleep(0.1)

    # -- understanding replies -------------------------------------------------

    def _interpret(self, url, status, content_type, body, polling):
        """Turn a response into one of:
            ("image", bytes)       the finished texture
            ("fetch", url)         the finished texture is at this URL
            ("poll", url, info)    not finished: check url again later
        """
        content_type = content_type.lower()
        if status >= 400:
            detail = _error_detail(body, content_type)
            if polling and status == 404:
                raise ServerError(f"The server no longer knows this job ({detail}). It may "
                                  "have been restarted; generate again")
            if polling and status in TRANSIENT_HTTP:
                raise TransientError(f"server error {status}: {detail}")
            raise ServerError(f"Server error {status}: {detail}")

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
        for key in ("image_url", "result_url"):
            if payload.get(key):
                return ("fetch", urllib.parse.urljoin(url, str(payload[key])))
        if state in DONE_STATES:
            raise ServerError("The server says the job is done, but its reply has no "
                              "'image', 'image_url' or 'result_url'")

        info = {
            "phase": "queued" if state in QUEUED_STATES else "running",
            "position": None,
            "progress": _number(payload.get("progress"), 0.0, 1.0),
            "message": str(payload.get("message") or ""),
        }
        position = _number(payload.get("position"), 0)
        if position is not None:
            info["position"] = int(position)
        if state and state not in QUEUED_STATES | RUNNING_STATES and not info["message"]:
            info["message"] = state                 # unknown state: show it as-is

        if payload.get("status_url"):
            return ("poll", urllib.parse.urljoin(url, str(payload["status_url"])), info)
        if polling and (state or status == 202):
            return ("poll", url, info)              # keep checking the same address
        if payload.get("job_id"):
            return ("poll", f"{self.server_url.rstrip('/')}/{payload['job_id']}", info)
        raise ServerError("The server's reply had no 'image', 'image_url', 'result_url' "
                          "or 'status_url'. See README.md for the expected format")

    # -- the whole conversation ------------------------------------------------

    def request_texture(self, image_png, prompt, face_size, faces, layout="horizontal_cross",
                        extra_fields=None):
        fields = {"prompt": prompt, "layout": layout, "face_size": str(face_size),
                  "faces": json.dumps(faces)}
        fields.update(extra_fields or {})
        body, content_type = encode_multipart(
            fields, {"image": ("cubemap.png", image_png, "image/png")})

        self._status(phase="uploading")
        # Not retried: if the upload half-succeeded, retrying could queue the job twice.
        # A server that answers synchronously may take the whole timeout to reply.
        first = self._interpret(self.server_url,
                                *self._send(self.server_url, body,
                                            {"Content-Type": content_type},
                                            request_timeout=self.timeout),
                                polling=False)
        return self._follow(first)

    def _follow(self, result):
        """Poll / download until we hold the image."""
        base = self.poll_interval
        interval = base
        last_info = None
        failures = 0
        while True:
            if self.is_cancelled():
                raise Cancelled()
            kind = result[0]
            if kind == "image":
                return result[1]

            if kind == "fetch":
                url = result[1]
                self._status(phase="downloading")
                step = lambda: ("image", self._download(url))   # noqa: E731
            else:
                _kind, url, info = result
                self.status_url = url
                if info["phase"] != "queued" and self.run_deadline is None:
                    self.run_deadline = time.monotonic() + self.timeout
                # Check often while running; while queued and nothing changes,
                # gradually slow down (up to 5x the interval) to go easy on the server.
                if info != last_info or info["phase"] != "queued":
                    interval = base
                else:
                    interval = min(interval * 1.5, base * 5)
                last_info = info
                self._status(**info)
                self._sleep(interval)
                step = lambda: self._interpret(                   # noqa: E731
                    url, *self._send(url, request_timeout=POLL_REQUEST_TIMEOUT), polling=True)

            while True:
                try:
                    result = step()
                    failures = 0
                    break
                except TransientError as err:
                    failures += 1
                    if failures > self.max_retries:
                        raise ServerError(f"Lost contact with the server: {err}")
                    delay = min(2 ** failures, 30)
                    self._status(phase="retrying", attempt=failures,
                                 max_attempts=self.max_retries, message=str(err))
                    self._sleep(delay)

    def _download(self, url):
        status, content_type, data = self._send(url, request_timeout=DOWNLOAD_TIMEOUT)
        if status >= 400:
            raise ServerError(f"Downloading the result failed ({status}): "
                              f"{_error_detail(data, content_type.lower())}")
        if not data:
            raise ServerError("The server sent an empty image")
        return data

    def cancel_remote(self):
        """Best effort: tell the server we gave up (DELETE on the status URL), so
        it can drop the job. Servers without that endpoint simply say 404/405."""
        if not self.status_url:
            return
        try:
            self._send(self.status_url, method="DELETE", request_timeout=5)
        except ServerError:
            pass
