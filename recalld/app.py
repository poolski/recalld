from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from recalld.config import DEFAULT_CONFIG_PATH, load_config, save_config
from recalld.jobs import DEFAULT_SCRATCH_ROOT
from recalld.runtime import cancel_pipeline_tasks

STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure required directories exist
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    # Ensure config file exists
    cfg = load_config()
    save_config(cfg)
    yield
    await cancel_pipeline_tasks()


def create_app() -> FastAPI:
    app = FastAPI(title="recalld", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    from recalld.routers import upload, jobs, settings, categories
    app.include_router(upload.router)
    app.include_router(jobs.router)
    app.include_router(settings.router)
    app.include_router(categories.router)

    return app


app = create_app()
