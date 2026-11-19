# UVUSee for Blender

Texture a mesh with AI from one prompt. The add-on renders the object from six
sides, sends the six views as one image to your own server, ChatGPT or Nano
Banana, and projects the answer back onto the object.

## Install

Blender 4.2 or newer: **Edit → Preferences → Get Extensions → ⌄ (top right) →
Install from Disk…** and choose the zip. Then open the add-on's preferences
(**Add-ons → UVUSee**), pick an **AI Provider** and fill in its details.

Unless the provider is a server on your own machine, enable **Preferences →
System → Network → Allow Online Access**. Requests to `localhost`/`127.0.0.1`
are always allowed.

## AI providers

Pick one in the add-on preferences or from the drop-down at the top of the panel.

| Provider | Needs | Default model |
|----------|-------|---------------|
| **Custom Server** | Server URL (+ optional API key) | your server (see *Server API* below) |
| **ChatGPT** | OpenAI API key | `gpt-image-2.5-sunburst` |
| **Nano Banana** | Gemini API key | `gemini-nano-banana-2.1` |

- If the key field is empty, `OPENAI_API_KEY` or `GEMINI_API_KEY` /
  `GOOGLE_API_KEY` from the environment is used.
- The model is a text field because both companies rename models often. For
  ChatGPT, use a model that accepts custom sizes (GPT Image 2 / 2.5), otherwise
  the 4:3 layout can't be requested.
- ChatGPT and Nano Banana are called directly from Blender, once per Generate,
  and billed to your key. There is no queue: the panel shows a timer until the
  image arrives. Busy/rate-limited answers (429/500/503) are retried twice.
- For these two, the views are sent on a plain black background (no
  transparency), an extra paragraph asks the model to keep every outline in
  place, and the result is stretched back to the capture's exact size.
- General image models redraw the whole picture, so the result can drift
  slightly from the silhouettes and the six views may not match at their
  edges. A custom server with a geometry-aware pipeline gives the most
  accurate alignment.

## Use

1. Select a mesh and open **Properties → Material tab → AI Texture**.
2. Type a prompt and press **Generate Texture**.
3. Keep working; the panel shows progress. When the result arrives it is
   applied automatically (one undo step), shown as a thumbnail, and packed into
   the .blend file.

Sub-sections:

- **Capture**: resolution per view, and *Clay* (shape only) or *Current Look*
  (the object's present colours, useful to refine an earlier result).
  *Preview Capture* makes the image without sending it.
- **Projection**: which side is the front, Lit/Flat material, edge padding,
  Keep Proportions.
- **Use Existing Cubemap**: project any cubemap image file yourself (cross,
  strip or 3×2 layouts, with per-face rotate/mirror fixes).

## Server API

### Request

`POST <Server URL>` as `multipart/form-data`:

| Field       | Type   | Content |
|-------------|--------|---------|
| `image`     | file   | `cubemap.png`, RGBA PNG, `4N × 3N` pixels |
| `prompt`    | text   | The user's prompt |
| `layout`    | text   | Always `horizontal_cross` |
| `face_size` | text   | `N`, pixels per face (256, 512, 1024 or 2048) |
| `faces`     | text   | JSON, `[column, row]` of each face, row 0 = top: `{"top":[1,0],"left":[0,1],"front":[1,1],"right":[2,1],"back":[3,1],"bottom":[1,2]}` |

Header `Authorization: Bearer <API Key>` is sent when a key is set.

The image is a horizontal cross:

```
           [ TOP  ]
  [ LEFT ] [FRONT ] [RIGHT ] [ BACK ]
           [BOTTOM]
```

Each face is an orthographic view of the object from outside, framed to the
object's bounding box on that side (so the object touches the edges of every
face). Transparent pixels (alpha 0) are background. Faces are oriented the way
the cross folds into a box with the pictures outside: the bottom edge of TOP
touches the top edge of FRONT, the left edge of RIGHT touches the right edge of
FRONT, and so on. Unused cells are transparent.

### Response

The texture must come back in the **same layout and orientation** (any size
with a 4:3 ratio, e.g. 2048×1536; other cubemap layouts are detected too). Each
face of the result is stretched over the matching side of the object, so keep
each face's content aligned with the silhouette in the request.

Answer with any **one** of:

1. **The image directly**: status 200, `Content-Type: image/png` (or
   `image/jpeg`, `image/webp`), body = image bytes.
2. **JSON with the image**: `{"image": "<base64>"}` (a `data:image/png;base64,`
   prefix is fine) or `{"image_url": "https://..."}` (absolute, or relative to
   the server URL; the API key is only sent if it's the same host).
3. **A background job** for slow generation: status 200 or 202 with
   `{"job_id": "abc", "status": "queued", "status_url": "/texture/abc"}`. The
   add-on then sends `GET status_url` regularly. If `status_url` is missing
   it polls `<Server URL>/<job_id>`. Each poll may answer:
   - waiting in a queue: `{"status": "queued", "position": 2}`. `position`
     (optional) is the number of jobs ahead; Blender shows "Queued: 2 jobs ahead"
   - working: `{"status": "running", "progress": 0.4, "message": "Sampling"}`.
     `progress` (0–1) shows as a progress bar, `message` next to the status
   - finished: `{"status": "done", "result_url": "/texture/abc/result"}`, or
     any of the forms in 1 or 2 (`result_url` and `image_url` mean the same)
   - failed: `{"status": "failed", "error": "Out of GPU memory"}`

   Status checks happen every 2 s while running (*Check Every* in the
   preferences). While a job sits in the queue and nothing changes, they
   slow down gradually to every 10 s.

Errors: any 4xx/5xx status is shown to the user, using the JSON `error`,
`detail` or `message` field if present, otherwise the start of the body.
While polling, a dropped connection or a 502/503/504 is retried up to 6
times with growing pauses (the job keeps running on the server), and a 404
means the server lost the job (e.g. it restarted).

The timeout in the preferences (default 300 s) limits how long a job may
**run**. Time waiting in the queue doesn't count; the user can press Cancel.

On Cancel the add-on sends `DELETE status_url` once, so a server that
supports it can drop the job. Servers without that endpoint can ignore it.

## Good to know

- The texture is applied through a UV map named `CubemapUV` and a material
  named `AI Texture <object>`; existing UV maps and materials are kept. Each
  new result for the same object replaces the previous image in that material.
- Projection is box mapping: a face gets the view its normal points to most.
  Surfaces hidden in every view (inside a cup, under an arm) get the colour of
  whatever covered them in that view. Slanted faces get stretched pixels.
- The bounding box covers modifiers too (e.g. Mirror), and capture and
  projection use the same box, so the result lines up with the views sent.
- Add-on preferences store the API key unencrypted.
