from pydantic import BaseModel, UUID4
from typing import List


class StudentInput(BaseModel):
    student_id: UUID4
    idade: int
    preco_plano: float
    meses_duracao_plano: int
    meses_desde_matricula: int
    media_dias_atraso: float
    historico_faturas_atrasadas: int
    total_faturas_pagas: int


class PredictRequest(BaseModel):
    students: List[StudentInput]


class PredictionResult(BaseModel):
    student_id: UUID4
    default_probability: float
    is_high_risk: bool


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
