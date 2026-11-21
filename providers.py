"""Hosted AI image models: ChatGPT (OpenAI GPT Image) and Nano Banana (Google
Gemini). The alternative to a custom server.

Pure Python standard library, no bpy: runs on the background thread, like
client.py, whose HTTP helper it reuses.

Both APIs are called once and answer when the image is ready (no queue), so
the panel shows "Generating with ChatGPT…" with a timer. A few things matter
for the result to line up with the object:
  - We ask for a picture with the same shape as the one we send. If the
    model can't make that shape, the capture is centred on a black canvas
    it can make (plan_canvas), and that area is cropped out of the answer.
  - The sent image has a plain background (no transparency) and the prompt
    asks the model to keep every outline in place.
  - jobs.py scales the result back to the exact size of the canvas and crops.
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
    "The picture shows these six views of one object on a plain black background. Keep "
    "the exact same layout, size, orientation and position of every view, and "
    "keep every outline and silhouette exactly where it is, so the result can be "
    "projected back onto the 3D object. Keep the background and all empty areas "
    "plain black. Do not add text, borders, shadows or new objects. "
)


def key_from_env(provider):
    for name in PROVIDERS[provider]["env"]:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


# Aspect ratios every current Nano Banana model accepts (Nano Banana 2 / 2.1
# also take 1:4, 4:1, 1:8, 8:1, but Pro and the original don't).
GEMINI_RATIOS = ((1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9))
OPENAI_MAX_PIXELS = 2048 * 1536        # keep requests at the 4:3 cross's usual size
OPENAI_MAX_SIDE = 3840


def _ceil_div(a, b):
    return -(-a // b)


def plan_canvas(provider, width, height):
    """How to send a width x height capture to a hosted model.

    Models only produce certain shapes, so the capture is centred on a black
    canvas with a shape they accept; jobs.py later crops the same area out of
    the answer. The 4:3 cross needs no padding with either provider; the
    proportional (Keep Proportions) layout often does.

    Returns {"canvas": (cw, ch, x, y), "upload_size": (w, h), ...} plus
    "openai_size" ("WxH") or "gemini_aspect" ("21:9") and "gemini_size".
    upload_size is what the canvas is scaled to before sending: the size the
    model will answer at, never larger than the canvas itself.
    """
    a = width / height
    if provider == "OPENAI":
        # Any size with sides divisible by 16 and an aspect between 1:3 and 3:1.
        cw, ch = width, height
        if a > 3:
            ch = _ceil_div(width, 3)
        elif a < 1 / 3:
            cw = _ceil_div(height, 3)
        scale = min(1.0, (OPENAI_MAX_PIXELS / (cw * ch)) ** 0.5, OPENAI_MAX_SIDE / max(cw, ch))
        ow = max(16, int(round(cw * scale / 16)) * 16)
        oh = max(16, int(round(ch * scale / 16)) * 16)
        # Rounding to 16 nudges the shape a little; pad the canvas to match it
        # exactly, so the answer can be scaled back without distortion.
        if cw / ch < ow / oh:
            cw = int(round(ch * ow / oh))
        else:
            ch = int(round(cw * oh / ow))
        plan = {"openai_size": f"{ow}x{oh}", "upload_size": (ow, oh)}
    else:
        best = None
        for rw, rh in GEMINI_RATIOS:
            r = rw / rh
            cw, ch = (_ceil_div(height * rw, rh), height) if r >= a else (width, _ceil_div(width * rh, rw))
            if best is None or cw * ch < best[0] * best[1]:
                best = (cw, ch, f"{rw}:{rh}")
        cw, ch, ratio = best
        # Gemini sizes: about 1 MP (1K), 4 MP (2K) or 16 MP (4K).
        area = cw * ch
        size = "1K" if area <= 1_200_000 else "2K" if area <= 4_600_000 else "4K"
        target = {"1K": 1_050_000, "2K": 4_200_000, "4K": 16_800_000}[size]
        shrink = min(1.0, (target / area) ** 0.5)
        plan = {"gemini_aspect": ratio, "gemini_size": size,
                "upload_size": (max(16, int(round(cw * shrink))), max(16, int(round(ch * shrink))))}
    plan["canvas"] = (cw, ch, (cw - width) // 2, (ch - height) // 2)
    return plan


def _api_error(body, content_type):
    """Both APIs answer errors as {"error": {"message": ..., ...}}. Keep the
    extra fields (OpenAI: param, code; Gemini: status): they say what to fix."""
    try:
        payload = json.loads(body)
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, dict) and err.get("message"):
            message = str(err["message"]).rstrip(". ")
            extra = [f"{key}: {err[key]}" for key in ("param", "code", "status")
                     if err.get(key) not in (None, "")]
            return f"{message} ({', '.join(extra)})" if extra else message
    except ValueError:
        pass
    return client._error_detail(body, content_type.lower())


class HostedClient(client.Client):
    """Calls OpenAI or Gemini. Reuses client.Client for HTTP, SSL and cancel."""

    def __init__(self, provider, api_key, model, timeout=300.0, quality="medium",
                 on_status=None, is_cancelled=None, upload_copy=""):
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
        self.upload_copy = upload_copy      # a copy of the image we send, for checking

    def generate(self, image_png, prompt, plan):
        """image_png: already padded to plan["canvas"] (see plan_canvas)."""
        self._status(phase="running", message=f"{self.label} ({self.model})", polled=False)
        try:
            if self.provider == "OPENAI":
                return self._openai(image_png, prompt, plan)
            return self._gemini(image_png, prompt, plan)
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
        elif "image" in detail.lower() and self.upload_copy:
            hint = hint or f"The image that was sent is saved at {self.upload_copy}"
        raise client.ServerError(f"{message}. {hint}" if hint else message)

    # -- OpenAI ------------------------------------------------------------------

    def _openai(self, image_png, prompt, plan):
        fields = {
            "model": self.model,
            "prompt": prompt,
            "size": plan["openai_size"],
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
                hint = ("This model may not support custom output sizes. Try "
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

    def _gemini(self, image_png, prompt, plan):
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
                "imageConfig": {"aspectRatio": plan["gemini_aspect"],
                                "imageSize": plan["gemini_size"]},
            },
        }
        status, ct, data = self._post(url, json.dumps(request).encode(), headers)
        if status == 400 and "imageconfig" in _api_error(data, ct).lower().replace("_", ""):
            # Older or newer models may not accept imageConfig; the result is
            # scaled to the canvas afterwards anyway, so try once without it.
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