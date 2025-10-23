# app/docs.py
from __future__ import annotations
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, Request
from .config import settings

# Se quiser YAML:
/* try: */
try:
    import yaml  # pip install pyyaml (opcional)
except Exception:  # pragma: no cover
    yaml = None


def swagger_ui_html(app: FastAPI) -> HTMLResponse:
    """
    Página Swagger customizada com cabeçalho (Env/Platform/Host/etc) e links rápidos,
    mantendo o bundle padrão via CDN.
    """
    # você pode dinamizar essas infos conforme seu ambiente
    ENV = "production"
    PLATFORM = "container"
    TIMEOUT = 15
    SUPABASE_URL = getattr(settings, "SUPABASE_URL", "-")
    HOST = settings.SERVICE_NAME

    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <title>{settings.SERVICE_NAME} — API Docs</title>
    <link rel="icon" href="https://fastapi.tiangolo.com/img/favicon.png">
    <style>
      body {{
        margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        background: #fff;
      }}
      .topbar {{ padding: 12px 16px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; }}
      .badges span {{
        display:inline-block; margin-right:8px; padding:4px 8px; border-radius:8px; background:#eef2ff; color:#3730a3; font-size:12px;
      }}
      .badges strong {{ color:#111827; }}
      .links a {{ margin-right:8px; text-decoration:none; background:#eff6ff; color:#1e40af; padding:4px 8px; border-radius:6px; font-size:12px; }}
      #swagger-ui {{ margin-top: 0; }}
    </style>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body>
    <div class="topbar">
      <div class="badges">
        <span><strong>Env</strong> {ENV}</span>
        <span><strong>Platform</strong> {PLATFORM}</span>
        <span><strong>Host</strong> {HOST}</span>
        <span><strong>Supabase URL</strong> {SUPABASE_URL}</span>
        <span><strong>Timeout</strong> {TIMEOUT}</span>
      </div>
      <div class="links" style="margin-top:8px;">
        <a href="/live" target="_blank">/live</a>
        <a href="/health" target="_blank">/health</a>
        <a href="/ready" target="_blank">/ready</a>
        <a href="/ops/status" target="_blank">/ops/status</a>
        <a href="/openapi.json" target="_blank">openapi.json</a>
        <a href="/openapi.yaml" target="_blank">openapi.yaml</a>
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
      }})
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
            # se PyYAML não estiver instalado, devolve JSON como texto
            import json
            return PlainTextResponse(json.dumps(schema, indent=2), media_type="application/yaml")
        return PlainTextResponse(yaml.safe_dump(schema, sort_keys=False), media_type="application/yaml")
