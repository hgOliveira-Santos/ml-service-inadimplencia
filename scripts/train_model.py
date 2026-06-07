import joblib
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Caminhos ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "mock"
MODELS_DIR = ROOT / "models"

# Features que serão utilizadas no modelo
FEATURES = [
    "idade",
    "preco_plano",
    "meses_duracao_plano",
    "meses_desde_matricula",
    "media_dias_atraso",
    "historico_faturas_atrasadas",
    "total_faturas_pagas",
]

REFERENCE_DATE = date(2026, 5, 30)
SEED = 42


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(data_dir: Path) -> pd.DataFrame:
    students    = pd.read_csv(data_dir / "students.csv",    parse_dates=["birth_date", "created_at"])
    plans       = pd.read_csv(data_dir / "plans.csv")
    enrollments = pd.read_csv(data_dir / "enrollments.csv", parse_dates=["start_date"])
    payments    = pd.read_csv(data_dir / "payments.csv",    parse_dates=["due_date", "paid_at"])

    ref = pd.Timestamp(REFERENCE_DATE)

    # Features demográficas e contratuais
    df = students[["id"]].copy()
    df["idade"]                = ((ref - students["birth_date"]).dt.days // 365).values
    df["meses_desde_matricula"] = ((ref - students["created_at"]).dt.days // 30).values

    enr = enrollments.merge(plans[["id", "price", "duration_months"]], left_on="plan_id", right_on="id")
    enr = enr.rename(columns={"price": "preco_plano", "duration_months": "meses_duracao_plano"})
    df = df.merge(enr[["student_id", "preco_plano", "meses_duracao_plano"]], left_on="id", right_on="student_id")

    # Features comportamentais de pagamento
    pay = payments.copy()
    pay["dias_atraso"] = (pay["paid_at"] - pay["due_date"]).dt.days.clip(lower=0)
    pay["is_late"]     = ((pay["status"] == "paid") & (pay["paid_at"] > pay["due_date"])) | (pay["status"] == "pending")

    late_mean  = pay[pay["is_late"]].groupby("student_id")["dias_atraso"].mean().rename("media_dias_atraso")
    late_count = pay.groupby("student_id")["is_late"].sum().rename("historico_faturas_atrasadas")
    paid_count = pay[pay["status"] == "paid"].groupby("student_id").size().rename("total_faturas_pagas")

    df = df.merge(late_mean,  left_on="id", right_index=True, how="left")
    df = df.merge(late_count, left_on="id", right_index=True, how="left")
    df = df.merge(paid_count, left_on="id", right_index=True, how="left")
    df = df.fillna({"media_dias_atraso": 0.0, "historico_faturas_atrasadas": 0, "total_faturas_pagas": 0})

    # Label: inadimplente se ≥ 30 % das faturas são problemáticas
    total_faturas = pay.groupby("student_id").size().rename("total_faturas")
    bad_faturas   = pay.groupby("student_id")["is_late"].sum().rename("bad_faturas")
    label_df      = (bad_faturas / total_faturas.clip(lower=1)).rename("inadimplente")
    label_df      = (label_df >= 0.30).astype(int)

    df = df.merge(label_df, left_on="id", right_index=True, how="left").fillna({"inadimplente": 0})
    df["inadimplente"] = df["inadimplente"].astype(int)

    return df


# ── Treinamento ───────────────────────────────────────────────────────────────

def main() -> None:
    print("Carregando e processando dados...")
    df = build_features(DATA_DIR)

    X = df[FEATURES].values
    y = df["inadimplente"].values

    print(f"  Dataset: {len(df)} alunos | Inadimplentes: {y.mean():.1%}\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ])

    print("Treinando modelo...")
    pipeline.fit(X_train, y_train)

    probas = pipeline.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, probas)
    preds  = (probas >= 0.70).astype(int)

    print(f"\n  AUC-ROC : {auc:.4f}")
    print("\n" + classification_report(y_test, preds, target_names=["adimplente", "inadimplente"]))

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "default_risk_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
