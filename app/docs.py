# app/docs.py
# =============================================================================
# Arquivo: docs.py
# Versão: v1
# Objetivo: Reconhecimento facial
# Funções/métodos:
# - endpoints do microserviço de reconhecimento facial no OpenIA e Swegger
# =============================================================================

from __future__ import annotations
import os
from typing import Iterable, Tuple
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI
from .config import settings

try:
    import yaml  # opcional (pyyaml)
except Exception:
    yaml = None

EXCLUDE_BADGES = {"/docs", "/openapi.json", "/openapi.yaml"}


def _collect_endpoints(app: FastAPI) -> Iterable[Tuple[str, str]]:
    """
    Retorna pares (method, path) únicos e ordenados, excluindo rotas internas de docs.
    """
    seen = set()
    items = []
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None)
        if not path or not methods or path in EXCLUDE_BADGES:
            continue
        for m in sorted(methods):
            mu = m.upper()
            if mu not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            key = (mu, path)
            if key not in seen:
                seen.add(key)
                items.append(key)
    items.sort(key=lambda x: (x[1], x[0]))
    return items


def swagger_ui_html(app: FastAPI) -> HTMLResponse:
    # --- Metadados dinâmicos (com fallbacks) ---
    SERVICE_NAME = getattr(
        settings, "SERVICE_NAME", getattr(app, "title", "svc-face-recon")
    )
    VERSION = getattr(settings, "SERVICE_VERSION", getattr(app, "version", "v1.0.0"))
    ENV = getattr(settings, "ENV", getattr(settings, "ENVIRONMENT", "production"))
    PLATFORM = "container"
    HOST = os.getenv("HOSTNAME", SERVICE_NAME)
    REDIS_STATUS = getattr(settings, "REDIS_STATUS", "ok")
    SUPABASE_URL = getattr(settings, "SUPABASE_URL", "-")
    RPC = getattr(settings, "DEFAULT_RPC", "-")
    TIMEOUT = getattr(settings, "REQUEST_TIMEOUT", 15)

    # Badges de endpoints (estilo bolhas à direita, como no svc-kg)
    ep_badges = [
        f'<a href="{p}" target="_self">{m} {p}</a>' for m, p in _collect_endpoints(app)
    ]
    endpoints_html = "\n        ".join(ep_badges)

    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>{SERVICE_NAME} — API Docs</title>
    <link rel="icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      :root {{
        --border:#e5e7eb;
        --muted:#6b7280;
        --bg:#ffffff;
        --tile:#f9fafb;
        --badge-bg:#eef2ff;
        --badge-fg:#3730a3;
        --link-bg:#eff6ff;
        --link-fg:#1e40af;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
        background: var(--bg);
        color: #111827;
      }}
      .header {{
        border-bottom: 1px solid var(--border);
        padding: 16px 24px;
        background: var(--bg);
      }}
      .title {{
        margin: 0 0 8px 0;
        font-size: 18px;
        font-weight: 600;
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(220px, 1fr));
        gap: 12px;
      }}
      @media (max-width: 1100px) {{
        .grid {{ grid-template-columns: repeat(2, minmax(200px, 1fr)); }}
      }}
      @media (max-width: 640px) {{
        .grid {{ grid-template-columns: 1fr; }}
      }}
      .card {{
        background: var(--tile);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 10px;
        font-size: 13px;
        line-height: 1.35;
      }}
      .card strong {{
        display: block;
        font-weight: 600;
        margin-bottom: 2px;
      }}
      .row-bottom {{
        margin-top: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
      }}
      .quick-links {{
        color: var(--muted);
        font-size: 13px;
      }}
      .quick-links a {{
        color: var(--link-fg);
        background: var(--link-bg);
        padding: 2px 6px;
        border-radius: 6px;
        text-decoration: none;
        margin-left: 8px;
        font-size: 12px;
      }}
      .badges a {{
        display: inline-block;
        text-decoration: none;
        margin: 4px 6px 0 0;
        padding: 4px 8px;
        border-radius: 999px;
        background: var(--badge-bg);
        color: var(--badge-fg);
        font-size: 12px;
        border: 1px solid #c7d2fe;
        white-space: nowrap;
      }}
      #swagger-ui {{ margin: 0; }}
      .swagger-ui .topbar {{ display:none; }}
    </style>
  </head>
  <body>
    <div class="header">
      <div class="title">{SERVICE_NAME} (a.k.a. “{SERVICE_NAME}”)</div>

      <div class="grid">
        <div class="card"><strong>Version</strong>{VERSION}</div>
        <div class="card"><strong>Env</strong>{ENV}</div>
        <div class="card"><strong>Platform</strong>{PLATFORM}</div>
        <div class="card"><strong>Host</strong>{HOST}</div>

        <div class="card"><strong>Redis</strong>{REDIS_STATUS}</div>
        <div class="card"><strong>Supabase URL</strong>{SUPABASE_URL}</div>
        <div class="card"><strong>RPC</strong>{RPC}</div>
        <div class="card"><strong>Timeout</strong>{TIMEOUT}</div>
      </div>

      <div class="row-bottom">
        <div class="quick-links">
          Links rápidos:
          <a href="/openapi.yaml" target="_blank">openapi.yaml</a>
          <a href="/openapi.json" target="_blank">openapi.json</a>
        </div>
        <div class="badges">
        {endpoints_html}
        </div>
      </div>
    </div>

    <div id="swagger-ui"></div>

    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
    <script>
      window.ui = SwaggerUIBundle({{
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        filter: true,
        layout: "BaseLayout",
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
        plugins: [SwaggerUIBundle.plugins.DownloadUrl],
        docExpansion: "none",
        defaultModelExpandDepth: 1,
        defaultModelsExpandDepth: 1,
        displayRequestDuration: true,
        tryItOutEnabled: true
      }});
    </script>
  </body>
</html>
"""
    return HTMLResponse(html)


def mount_docs_routes(app: FastAPI) -> None:
    @app.get("/docs", include_in_schema=False)
    async def custom_docs() -> HTMLResponse:
        return swagger_ui_html(app)

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_json() -> dict:
        return get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
            description=app.description,
            tags=app.openapi_tags,
        )

    @app.get("/openapi.yaml", include_in_schema=False)
    async def openapi_yaml() -> PlainTextResponse:
        schema = get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
            description=app.description,
            tags=app.openapi_tags,
        )
        if yaml is None:
            import json

            return PlainTextResponse(
                json.dumps(schema, indent=2), media_type="application/yaml"
            )
        return PlainTextResponse(
            yaml.safe_dump(schema, sort_keys=False), media_type="application/yaml"
        )
