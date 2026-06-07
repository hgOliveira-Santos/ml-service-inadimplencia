import numpy as np
from fastapi import APIRouter, Request
from api.schemas import PredictRequest, PredictResponse, PredictionResult

router = APIRouter()

FEATURES = [
    "idade",
    "preco_plano",
    "meses_duracao_plano",
    "meses_desde_matricula",
    "media_dias_atraso",
    "historico_faturas_atrasadas",
    "total_faturas_pagas",
]

HIGH_RISK_THRESHOLD = 0.70


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest):
    model = request.app.state.model

    student_ids = [s.student_id for s in body.students]
    X = np.array([[getattr(s, f) for f in FEATURES] for s in body.students])

    probabilities = model.predict_proba(X)[:, 1]

    predictions = [
        PredictionResult(
            student_id=sid,
            default_probability=round(float(prob), 4),
            is_high_risk=float(prob) >= HIGH_RISK_THRESHOLD,
        )
        for sid, prob in zip(student_ids, probabilities)
    ]

    return PredictResponse(predictions=predictions)
