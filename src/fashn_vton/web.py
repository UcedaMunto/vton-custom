"""Public web frontend (F6 of ``kubernetes/00_PLAN_ARQUITECTURA.md``).

A tiny navigable page for the published service: upload person + garment, follow
the job and download the result. Design rules from the plan (section 10):

* It loads **no model**, touches no GPU and never talks to Redis/MinIO: it only
  calls ``vton-api-gateway`` over the cluster network.
* The gateway API key lives in a Secret and is added **server-side**, so the
  browser never sees it.
* Per-IP rate limit and a signed session cookie (no personal data in it), plus a
  visible privacy notice: the images are transient and are deleted (section 11).

Run it with ``uvicorn fashn_vton.web:app``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool


def _package_version() -> str:
    """Version del paquete SIN importar ``fashn_vton`` (cuyo __init__ carga torch).

    El gateway y el frontend no necesitan torch ni el pipeline: importar el
    paquete completo costaba ~300 MB de RSS (el frontend moria con OOMKilled con
    un limite de 256Mi, ver 02_ESTADO_DE_EJECUCION.md).
    """
    try:
        from importlib.metadata import version

        return version("fashn-vton")
    except Exception:  # noqa: BLE001 - el paquete puede no estar instalado como distribuible
        return "unknown"


__version__ = _package_version()

logger = logging.getLogger("fashn_vton.web")

GATEWAY_URL = os.environ.get("VTON_GATEWAY_URL", "http://127.0.0.1:8000").rstrip("/")
GATEWAY_KEY = os.environ.get("VTON_GATEWAY_API_KEY", "")
SESSION_SECRET = os.environ.get("VTON_SESSION_SECRET", "")
MAX_UPLOAD_MB = int(os.environ.get("VTON_MAX_UPLOAD_MB", "12"))
RATE_LIMIT_PER_10MIN = int(os.environ.get("VTON_RATE_LIMIT_PER_10MIN", "6"))
RETENTION_MINUTES = int(os.environ.get("VTON_RETENTION_MINUTES", "30"))
PUBLIC_HOST = os.environ.get("VTON_PUBLIC_HOST", "")
ALLOW_ANONYMOUS = os.environ.get("VTON_ALLOW_ANONYMOUS", "0").strip().lower() in {"1", "true", "yes", "on"}

app = FastAPI(title="vton-web", version=__version__, description="Frontend publico del servicio de try-on")


def _session_token() -> str:
    if not SESSION_SECRET:
        return ""
    stamp = str(int(time.time()))
    mac = hmac.new(SESSION_SECRET.encode(), stamp.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{stamp}.{mac}"


def _session_ok(token: str | None) -> bool:
    if not SESSION_SECRET or not token or "." not in token:
        return True  # sin secreto configurado no se exige sesion (modo laboratorio)
    stamp, mac = token.split(".", 1)
    expected = hmac.new(SESSION_SECRET.encode(), stamp.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(mac, expected):
        return False
    try:
        return (time.time() - int(stamp)) < 6 * 3600
    except ValueError:
        return False


def _gateway_headers() -> dict:
    headers = {"Accept": "application/json"}
    if GATEWAY_KEY:
        headers["X-API-Key"] = GATEWAY_KEY
    return headers


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Encode a multipart/form-data body with the standard library only."""
    boundary = f"----vton{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
    for name, (filename, payload, content_type) in files.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
        body += payload + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _gateway_request(path: str, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None, timeout: int = 30):
    headers = _gateway_headers()
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(f"{GATEWAY_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL fija del cluster
        payload = response.read()
        ctype = response.headers.get("Content-Type", "")
    if "application/json" in ctype:
        return json.loads(payload.decode("utf-8"))
    return payload


def _rate_ok(request: Request) -> bool:
    """Per-IP quota; uses Redis when the frontend is given a REDIS_URL."""
    url = os.environ.get("REDIS_URL")
    if not url or not RATE_LIMIT_PER_10MIN:
        return True
    try:
        from .cluster import bus

        client_ip = (request.client.host if request.client else "?") or "?"
        return bus.allow_rate(bus.connect(), f"web:{client_ip}", RATE_LIMIT_PER_10MIN, 600)
    except Exception as exc:  # noqa: BLE001 - never block the service on a quota check
        logger.warning("No se pudo comprobar la cuota por IP: %s", exc)
        return True


@app.get("/healthz")
def healthz() -> dict:
    payload = {
        "status": "ok",
        "component": "vton-web",
        "package_version": __version__,
        "gateway": GATEWAY_URL,
        "session_cookie": bool(SESSION_SECRET),
        "public_host": PUBLIC_HOST,
        "retention_minutes": RETENTION_MINUTES,
    }
    try:
        payload["gateway_health"] = _gateway_request("/healthz", timeout=5)
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "degraded"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    return payload


@app.post("/submit")
async def submit(
    request: Request,
    person: Annotated[UploadFile, File()],
    garment: Annotated[UploadFile, File()],
    category: Annotated[str, Form()] = "tops",
    garment_photo_type: Annotated[str, Form()] = "flat-lay",
    num_timesteps: Annotated[int, Form()] = 30,
    seed: Annotated[int, Form()] = 42,
) -> JSONResponse:
    if not _session_ok(request.cookies.get("vton_session")):
        raise HTTPException(status_code=403, detail="Sesion no valida; recarga la pagina.")
    if not await run_in_threadpool(_rate_ok, request):
        raise HTTPException(status_code=429, detail=f"Has superado la cuota ({RATE_LIMIT_PER_10MIN} cada 10 min). Prueba mas tarde.")

    person_raw = await person.read()
    garment_raw = await garment.read()
    for raw, role in ((person_raw, "persona"), (garment_raw, "prenda")):
        if not raw:
            raise HTTPException(status_code=422, detail=f"Falta la imagen de {role}.")
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"La imagen de {role} supera {MAX_UPLOAD_MB} MB.")

    fields = {
        "category": category,
        "garment_photo_type": garment_photo_type,
        "num_timesteps": num_timesteps,
        "seed": seed,
    }
    body, content_type = _multipart(fields, {
        "person": (person.filename or "person.jpg", person_raw, person.content_type or "application/octet-stream"),
        "garment": (garment.filename or "garment.jpg", garment_raw, garment.content_type or "application/octet-stream"),
    })
    try:
        result = await run_in_threadpool(_gateway_request, "/v1/jobs", data=body, content_type=content_type, timeout=60)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("No se pudo encolar el job")
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {type(exc).__name__}: {exc}") from exc

    response = JSONResponse(content=result, status_code=202)
    if SESSION_SECRET:
        response.set_cookie("vton_session", _session_token(), httponly=True, samesite="lax", max_age=6 * 3600)
    return response


@app.get("/jobs/{job_id}")
def job_state(job_id: str) -> dict:
    try:
        return _gateway_request(f"/v1/jobs/{job_id}", timeout=10)
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace")) from exc


@app.get("/jobs/{job_id}/image")
def job_image(job_id: str) -> Response:
    """Proxy the result so the gateway can delete it on delivery."""
    try:
        data = _gateway_request(f"/v1/jobs/{job_id}/result", timeout=60)
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=exc.read().decode("utf-8", "replace")) from exc
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "no-store"})


PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Probador virtual</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0 auto; max-width: 780px; padding: 24px; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }}
 fieldset {{ border: 1px solid #ddd; border-radius: 8px; margin-bottom: 16px; }}
 label {{ display: block; margin: 8px 0 4px; font-weight: 600; }}
 input, select, button {{ font: inherit; padding: 6px; }}
 button {{ background: #1864ab; color: #fff; border: 0; border-radius: 6px; padding: 10px 18px; cursor: pointer; }}
 button[disabled] {{ opacity: .6; cursor: default; }}
 #log {{ background: #f8f9fa; border-radius: 6px; padding: 10px; white-space: pre-wrap; font-size: .9rem; }}
 #result {{ max-width: 100%; border-radius: 8px; margin-top: 12px; }}
 .aviso {{ background: #fff3bf; border-radius: 6px; padding: 10px; font-size: .85rem; }}
</style>
</head>
<body>
<h1>Probador virtual (try-on)</h1>
<p class="aviso">Tus imagenes se procesan de forma <strong>efimera</strong>: se borran del servidor
en cuanto se generan/descargan (retencion maxima {retencion} min) y no se usan para entrenar ningun modelo.</p>
<form id="form">
  <fieldset><legend>Persona (foto de cuerpo completo)</legend>
  <input type="file" name="person" accept="image/*" required></fieldset>
  <fieldset><legend>Prenda</legend>
  <input type="file" name="garment" accept="image/*" required>
  <label>Como es la foto de la prenda</label>
  <select name="garment_photo_type"><option value="flat-lay">Prenda estirada (producto)</option><option value="model">Prenda puesta por una modelo</option></select>
  </fieldset>
  <fieldset><legend>Parametros</legend>
  <label>Categoria</label>
  <select name="category"><option value="tops">Parte de arriba</option><option value="bottoms">Parte de abajo</option><option value="one-pieces">Vestido / mono</option></select>
  <label>Pasos de difusion (mas = mejor y mas lento)</label>
  <input type="number" name="num_timesteps" value="30" min="1" max="50">
  <label>Semilla</label>
  <input type="number" name="seed" value="42">
  </fieldset>
  <button type="submit" id="go">Generar</button>
</form>
<div id="log">Listo.</div>
<img id="result" alt="" hidden>
<script>
const form = document.getElementById('form');
const log = (m) => {{ document.getElementById('log').textContent = m; }};
form.addEventListener('submit', async (event) => {{
  event.preventDefault();
  const go = document.getElementById('go'); go.disabled = true;
  document.getElementById('result').hidden = true;
  try {{
    log('Subiendo imagenes...');
    const job = await (await fetch('/submit', {{ method: 'POST', body: new FormData(form) }})).json();
    if (job.detail) throw new Error(JSON.stringify(job.detail));
    log('Job ' + job.job_id + ' en cola. Esperando turno de GPU...');
    let state = {{}};
    for (let i = 0; i < 240; i++) {{
      await new Promise((r) => setTimeout(r, 3000));
      state = await (await fetch('/jobs/' + job.job_id)).json();
      log('Estado: ' + state.status + ' (' + (state.progress || 0) + '%)' + (state.error ? ' - ' + state.error : ''));
      if (state.status === 'done' || state.status === 'error') break;
    }}
    if (state.status === 'done') {{
      const img = document.getElementById('result');
      img.src = '/jobs/' + job.job_id + '/image';
      img.hidden = false;
      log('Listo. La imagen se envia una sola vez y se borra del servidor.');
    }} else if (state.status !== 'done') {{
      log('No se pudo completar: ' + (state.error || state.status));
    }}
  }} catch (err) {{
    log('Error: ' + err.message);
  }} finally {{
    go.disabled = false;
  }}
}});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE.format(retencion=RETENTION_MINUTES))


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    import uvicorn

    uvicorn.run("fashn_vton.web:app", host="0.0.0.0", port=int(os.environ.get("WEB_PORT", "8080")))
