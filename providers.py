"""Hosted AI image models: ChatGPT (OpenAI GPT Image) and Nano Banana (Google
Gemini). The alternative to a custom server.

Pure Python standard library, no bpy: runs on the background thread, like
client.py, whose HTTP helper it reuses.

Both APIs are called once and answer when the image is ready (no queue), so
the panel shows "Generating with ChatGPT…" with a timer. A few things matter
for the result to line up with the object:
  - We ask for a 4:3 picture, the shape of the cross we send.
  - The sent image has a plain background (no transparency) and the prompt
    asks the model to keep every outline in place.
  - jobs.py stretches the result back to the exact size of the capture.
"""

import base64
import json
import os
import time

from . import client

OPENAI_EDIT_URL = "https://api.openai.com/v1/images/edits"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROVIDERS = {
    "OPENAI": {
        "label": "ChatGPT",
        "long": "ChatGPT (OpenAI GPT Image)",
        "env": ("OPENAI_API_KEY",),
        "key_page": "https://platform.openai.com/api-keys",
        # GPT Image 2.5 accepts custom sizes such as 2048x1536 (4:3); OpenAI
        # recommends Sunburst for precise edits, Flare for speed.
        "default_model": "gpt-image-2.5-sunburst",
    },
    "GEMINI": {
        "label": "Nano Banana",
        "long": "Nano Banana (Google Gemini)",
        "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "key_page": "https://aistudio.google.com/apikey",
        "default_model": "gemini-nano-banana-2.1",
    },
}

RETRY_STATUSES = {429, 500, 503}   # rate limited / overloaded: worth another go
RETRY_DELAYS = (5, 15)             # seconds; two retries at most

# Added between the user's context prompt and their own words. Hosted models
# don't know they're painting a cubemap, so spell out what must not move.
ALIGNMENT_PROMPT = (
    "The picture is a 4 by 3 grid holding six views of one object on a plain black "
    "background. Keep the exact same layout, size and position of every view, and "
    "keep every outline and silhouette exactly where it is, so the result can be "
    "projected back onto the 3D object. Keep the background and the empty grid cells "
    "plain black. Do not add text, borders, shadows or new objects. "
)


def key_from_env(provider):
    for name in PROVIDERS[provider]["env"]:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def output_size_openai(width, height):
    """Largest 4:3 size, multiple of 16, up to the capture size and 2048x1536."""
    scale = min(1.0, 2048 / width, 1536 / height)
    w = max(16, int(width * scale) // 16 * 16)
    h = max(16, int(height * scale) // 16 * 16)
    return f"{w}x{h}"


def image_size_gemini(width):
    """Gemini's 4:3 outputs are 1200x896 (1K), 2400x1792 (2K), 4800x3584 (4K)."""
    if width <= 1200:
        return "1K"
    if width <= 2400:
        return "2K"
    return "4K"


def _api_error(body, content_type):
    """Both APIs answer errors as {"error": {"message": ...}}."""
    try:
        payload = json.loads(body)
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    except ValueError:
        pass
    return client._error_detail(body, content_type.lower())


class HostedClient(client.Client):
    """Calls OpenAI or Gemini. Reuses client.Client for HTTP, SSL and cancel."""

    def __init__(self, provider, api_key, model, timeout=300.0, quality="medium",
                 on_status=None, is_cancelled=None):
        base = OPENAI_EDIT_URL if provider == "OPENAI" else GEMINI_URL.format(model=model)
        # OpenAI uses "Authorization: Bearer", which client.Client adds for us;
        # Gemini uses its own header, added in _gemini.
        super().__init__(base, api_key if provider == "OPENAI" else "",
                         timeout=timeout, on_status=on_status, is_cancelled=is_cancelled)
        self.provider = provider
        self.key = api_key
        self.model = model
        self.quality = quality
        self.label = PROVIDERS[provider]["label"]

    def generate(self, image_png, prompt, width, height):
        self._status(phase="running", message=f"{self.label} ({self.model})", polled=False)
        try:
            if self.provider == "OPENAI":
                return self._openai(image_png, prompt, width, height)
            return self._gemini(image_png, prompt, width, height)
        except client.TransientError as err:
            raise client.ServerError(f"Couldn't get an answer from {self.label}: {err}")

    # -- shared ----------------------------------------------------------------

    def _post(self, url, body, headers):
        """POST with retries on 429/500/503. Network failures are NOT retried:
        the request may have gone through and been billed."""
        for attempt in range(len(RETRY_DELAYS) + 1):
            if self.is_cancelled():
                raise client.Cancelled()
            status, content_type, data = self._send(url, body, headers,
                                                    request_timeout=self.timeout)
            if status in RETRY_STATUSES and attempt < len(RETRY_DELAYS):
                delay = RETRY_DELAYS[attempt]
                self._status(phase="retrying", attempt=attempt + 1,
                             max_attempts=len(RETRY_DELAYS),
                             message=f"{self.label} is busy ({status}): "
                                     f"{_api_error(data, content_type)}")
                self._sleep(delay)
                self._status(phase="running", message=f"{self.label} ({self.model})",
                             polled=False)
                continue
            return status, content_type, data
        return status, content_type, data

    def _fail(self, status, content_type, data, hint=""):
        detail = _api_error(data, content_type)
        message = f"{self.label} error {status}: {detail.rstrip('. ')}"
        if status in (401, 403) or "api key" in detail.lower():
            hint = hint or "Check the API key in the add-on preferences"
        raise client.ServerError(f"{message}. {hint}" if hint else message)

    # -- OpenAI ------------------------------------------------------------------

    def _openai(self, image_png, prompt, width, height):
        fields = {
            "model": self.model,
            "prompt": prompt,
            "size": output_size_openai(width, height),
            "quality": self.quality,
            "background": "opaque",
            "output_format": "png",
            "n": "1",
        }
        body, content_type = client.encode_multipart(
            fields, {"image": ("cubemap.png", image_png, "image/png")})
        status, ct, data = self._post(OPENAI_EDIT_URL, body, {"Content-Type": content_type})
        if status >= 400:
            detail = _api_error(data, ct).lower()
            hint = ""
            if "size" in detail:
                hint = ("This model may not support 4:3 output sizes. Try "
                        "gpt-image-2.5-sunburst or gpt-image-2.5-flare")
            elif "model" in detail:
                hint = "Check the model name in the add-on preferences"
            self._fail(status, ct, data, hint)
        try:
            payload = json.loads(data)
            return base64.b64decode(payload["data"][0]["b64_json"])
        except (ValueError, KeyError, IndexError, TypeError):
            raise client.ServerError("ChatGPT's reply had no image in it")

    # -- Gemini ------------------------------------------------------------------

    def _gemini(self, image_png, prompt, width, height):
        url = GEMINI_URL.format(model=self.model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.key}
        request = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png",
                                     "data": base64.b64encode(image_png).decode("ascii")}},
                    {"text": prompt},
                ],
            }],
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": "4:3", "imageSize": image_size_gemini(width)},
            },
        }
        status, ct, data = self._post(url, json.dumps(request).encode(), headers)
        if status == 400 and "imageconfig" in _api_error(data, ct).lower().replace("_", ""):
            # Older or newer models may not accept imageConfig; the result is
            # stretched to 4:3 afterwards anyway, so try once without it.
            del request["generationConfig"]["imageConfig"]
            status, ct, data = self._post(url, json.dumps(request).encode(), headers)
        if status >= 400:
            hint = "Check the model name in the add-on preferences" if status == 404 else ""
            self._fail(status, ct, data, hint)

        try:
            payload = json.loads(data)
        except ValueError:
            raise client.ServerError("Nano Banana's reply wasn't valid JSON")
        texts = []
        for candidate in payload.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                blob = part.get("inlineData") or part.get("inline_data")
                if blob and blob.get("data"):
                    return base64.b64decode(blob["data"])
                if part.get("text"):
                    texts.append(part["text"].strip())

        # No image: explain why if the API says.
        reason = (payload.get("promptFeedback") or {}).get("blockReason")
        if not reason and payload.get("candidates"):
            reason = payload["candidates"][0].get("finishReason")
        message = "Nano Banana returned no image"
        if reason:
            message += f" (reason: {reason})"
        if texts:
            message += f". It said: \"{' '.join(texts)[:200]}\""
        raise client.ServerError(message)
