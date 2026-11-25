# UVUSee for Blender

Texture a mesh with AI from one prompt. The add-on renders the object from six
sides, sends the six views as one image to your own server, ChatGPT or Nano
Banana, and projects the answer back onto the object.

## Install

Blender 4.2 or newer: **Edit → Preferences → Get Extensions → (top right) →
Install from Disk…** and choose the zip. Then open the add-on's preferences
(**Add-ons → UVUSee**), pick an **AI Provider** and fill in its details.

Unless the provider is a server on your own machine, enable **Preferences →
System → Network → Allow Online Access**. Requests to `localhost`/`127.0.0.1`
are always allowed.

## AI providers

Pick one in the add-on preferences or from the drop-down at the top of the panel.
**You will need to add a minimum credit in order to use APIs from ChatGPT and Gemini on their website.**

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
  place, and the result is scaled back to the capture's exact size. If the
  capture's shape isn't one the model can produce (e.g. a very wide *Keep
  Proportions* layout), it is centred on a larger black canvas that is, and
  the same area is cropped out of the answer.
- General image models redraw the whole picture, so the result can drift
  slightly from the silhouettes and the six views may not match at their
  edges. A custom server with a geometry-aware pipeline gives the most
  accurate alignment.

## Custom Server
To use the custom server, please head on to [uvusee-server](https://github.com/AeroAndZero/uvusee-server) repo and read the README

## Use

1. Select a mesh and open **Properties → Material tab → AI Texture**.
2. Type a prompt and press **Generate Texture**.
3. Keep working; the panel shows progress. When the result arrives it is
   applied automatically (one undo step), shown as a thumbnail, and packed into
   the .blend file.

Sub-sections:

- **Capture**: the layout of the six views in the image sent to the AI
  (horizontal cross by default; vertical cross, strips or a 3×2 grid),
  resolution per view, and *Clay* (shape only) or *Current Look*
  (the object's present colours, useful to refine an earlier result).
  *Preview Capture* makes the image without sending it.
- **Projection**: the method (below), which side is the front, Lit/Flat
  material, edge padding, Keep Proportions.
- **Use Existing Cubemap**: project any cubemap image file yourself (cross,
  strip or 3×2 layouts, with per-face rotate/mirror fixes). Its *Image Layout*
  only describes that file; it doesn't change the AI capture.

## Projection methods

- **Blend & Bake** (default): bakes all views into one UV texture. Each spot
  takes a blend of the views that see it, weighted towards the one looking at
  it most directly, so seams are soft and slanted faces aren't stretched.
  Areas no view saw are filled from their surroundings. Options: texture size,
  UVs (*Auto Unwrap* into a `UVUSeeBake` UV map, or your *Active UV Map*),
  *Blend Sharpness* and *Max View Angle*. How it works: at capture time each
  view is rendered a second time as a hidden "lookup" image holding, per
  pixel, the UV coordinate of the visible surface and how directly the view
  sees it; when the painted views return, every painted pixel is added into
  the texture at its UV (`bake.py`). The bake runs in the background and takes
  a few seconds (longer at 4096 px).
- **Box**: every face takes the one view it points at most, through a
  `CubemapUV` map that reads the returned image directly. Instant, but with
  hard seams and stretching on slanted faces.

## Server API

### Request

`POST <Server URL>` as `multipart/form-data`:

| Field       | Type   | Content |
|-------------|--------|---------|
| `image`     | file   | `cubemap.png`, RGBA PNG, `4N × 3N` pixels |
| `prompt`    | text   | The user's prompt |
| `layout`    | text   | `horizontal_cross` (default), `vertical_cross`, `horizontal_strip`, `vertical_strip` or `grid_3x2` (*Capture > Layout*); with *Keep Proportions*: `proportional_cross` for the horizontal cross, otherwise `proportional_<layout>` |
| `face_size` | text   | `N`, pixels per face (256, 512, 1024 or 2048) |
| `faces`     | text   | JSON, `[column, row]` of each face, row 0 = top: `{"top":[1,0],"left":[0,1],"front":[1,1],"right":[2,1],"back":[3,1],"bottom":[1,2]}` (for `proportional_cross`: pixel rectangles, as in `face_rects`) |
| `face_rects` | text  | JSON, `[x, y, width, height]` of each face in pixels, top-left origin, for any layout |
| `face_turns` | text  | JSON, quarter turns counter-clockwise each face is stored with, e.g. `{"back": 2, ...}` (only the vertical cross's back is turned) |

Header `Authorization: Bearer <API Key>` is sent when a key is set.

By default the image is a horizontal cross:

```
           [ TOP  ]
  [ LEFT ] [FRONT ] [RIGHT ] [ BACK ]
           [BOTTOM]
```

Other layouts (*Capture > Layout*): vertical cross (top / left front right /
bottom / back, the back turned upside down), horizontal or vertical strip
(right, left, top, bottom, front, back) and 3×2 grid (right left top / bottom
front back). `faces`, `face_rects` and `face_turns` always say exactly where
each face is. The answer must use the same layout and shape as the request.

With **Keep Proportions** on, the layout is `proportional_cross`: the same
cross, but each cell has the true proportions of that side of the object's
bounding box (width × depth × height), all at the same pixel density. The image
is then not 4:3 (a 4 × 1 × 1 plank gives roughly 10:3) and its width and height
are multiples of 16; a strip on the right/bottom may be empty. Use `face_rects`
to find each face. The result must have the **same proportions** as the sent
image (any resolution).

Each face is an orthographic view of the object from outside, framed to the
object's bounding box on that side (so the object touches the edges of every
face). Transparent pixels (alpha 0) are background. Faces are oriented the way
the cross folds into a box with the pictures outside: the bottom edge of TOP
touches the top edge of FRONT, the left edge of RIGHT touches the right edge of
FRONT, and so on. Unused cells are transparent.

### Response

The texture must come back in the **same layout and orientation**, at any
size with the same proportions (e.g. 2048×1536 for the 4:3 cross; for the 4:3
cross other cubemap layouts are detected too). Each
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

   Status checks happen every 5 seconds while running (*Check Every* in the
   preferences). While a job sits in the queue and nothing changes, they
   slow down gradually to every 10 seconds.

Errors: any 4xx/5xx status is shown to the user, using the JSON `error`,
`detail` or `message` field if present, otherwise the start of the body.
While polling, a dropped connection or a 502/503/504 is retried up to 6
times with growing pauses (the job keeps running on the server), and a 404
means the server lost the job (e.g. it restarted).

As long as the server keeps answering status checks with `queued` or
`running`, the add-on keeps waiting, however long the job takes; the user can
press Cancel at any time. *Response Timeout* (default 300 s) only applies to a
single request that gets no answer (e.g. the upload, or a server that answers
synchronously). *Max Job Time* (default 0 = no limit) can cap how long a job
may run once the server reports it as `running`.

On Cancel the add-on sends `DELETE status_url` once, so a server that
supports it can drop the job. Servers without that endpoint can ignore it.

## Just a heads up

- The texture is applied through a material named `AI Texture <object>` and a
  UV map (`UVUSeeBake` or your own for Blend & Bake, `CubemapUV` for Box);
  existing UV maps and materials are kept. Each new result for the same object
  replaces the previous image in that material.
- With six straight-on views, surfaces hidden in all of them (inside a cup,
  under an arm) are guessed: Blend & Bake fills them from their surroundings,
  Box gives them the colour of whatever covered them.
- The bounding box covers modifiers too (e.g. Mirror), and capture and
  projection use the same box, so the result lines up with the views sent.
- Add-on preferences store the API key unencrypted.

## Roadmap
- [ ] Add ControlNet workflow to accommodate Depth texture and Normal textures
- [ ] Add inpainting support
- [ ] Add batched image generation mode

## ❤️ Respsect the artists ❤️
Texturing is a big part of any game development. Texture artists are irreplacable. This add-on is for indie developers who wants to speed up their process and experiment with different type of textures. 

This add-on will not produce realistic and accurate results; and I would like to keep it that way.