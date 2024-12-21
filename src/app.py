from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from pathlib import Path
from src.api_router.model import router 
# Define the base directory for the app folder
#BASE_DIR = Path(__file__).resolve().parent
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/data", response_class=HTMLResponse)
async def read_data(request: Request):
    return templates.TemplateResponse("components/data.html", {"request": request})

@app.get("/model", response_class=HTMLResponse)
async def read_model(request: Request):
    return templates.TemplateResponse("components/model.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def read_predict(request: Request):
    return templates.TemplateResponse("components/predict.html", {"request": request})

app.include_router(router)