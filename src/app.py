from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates setup
templates = Jinja2Templates(directory="templates")

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
