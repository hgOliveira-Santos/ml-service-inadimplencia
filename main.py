import joblib
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from api.routes import router

MODEL_PATH = Path(__file__).parent / "models" / "default_risk_model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="ML Service — Risco de Inadimplência", version="1.0.0", lifespan=lifespan)
app.include_router(router)


@app.get("/")
def health():
    return {"status": "running"}
