import uuid
import random
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

fake = Faker("pt_BR")

# ── Configurações ────────────────────────────────────────────────────────────
NUM_STUDENTS   = 2_000
REFERENCE_DATE = date(2026, 5, 30)          
SIM_START_DATE = date(2023, 1, 1)           
DATA_DIR       = Path(__file__).resolve().parent.parent / "data" / "mock"

PLANS = [
    {"id": str(uuid.uuid4()), "name": "Básico",  "price":  50.0, "duration_months": 12},
    {"id": str(uuid.uuid4()), "name": "Padrão",  "price": 100.0, "duration_months": 12},
    {"id": str(uuid.uuid4()), "name": "Premium", "price": 200.0, "duration_months": 12},
]
_PLAN_BY_NAME = {p["name"]: p for p in PLANS}

_PLAN_WEIGHTS = [0.40, 0.35, 0.25]


# ── Amostragem de idade com distribuição realista ────────────────────────────

def _sample_age() -> int:
    """
    Distribui idades em baldes com pesos para reproduzir uma base de alunos real.
    O balde 18–23 representa jovens adultos com maior volatilidade financeira.
    """
    buckets = [
        (range(18, 24), 0.25),
        (range(24, 36), 0.40),
        (range(36, 46), 0.25),
        (range(46, 65), 0.10),
    ]
    chosen = random.choices(
        [r for r, _ in buckets],
        weights=[w for _, w in buckets],
    )[0]
    return random.choice(list(chosen))


# ── Geração de tabelas ───────────────────────────────────────────────────────

def make_students(n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        age        = _sample_age()
        birth_date = REFERENCE_DATE - timedelta(days=age * 365 + random.randint(0, 364))
        created_at = fake.date_between(start_date=SIM_START_DATE, end_date=REFERENCE_DATE - timedelta(days=30))
        rows.append({
            "id":         str(uuid.uuid4()),
            "name":       fake.name(),
            "birth_date": birth_date.isoformat(),
            "created_at": created_at.isoformat(),
        })
    return pd.DataFrame(rows)


def make_enrollments(students: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Retorna (DataFrame de matrículas, mapa {student_id: nome_do_plano}).
    O mapa evita um join posterior no loop de pagamentos.
    """
    rows: list[dict] = []
    plan_map: dict[str, str] = {}

    for _, s in students.iterrows():
        plan       = random.choices(PLANS, weights=_PLAN_WEIGHTS)[0]
        created_at = date.fromisoformat(s["created_at"])
        # Matrícula ocorre até 7 dias após o cadastro
        start_date = created_at + timedelta(days=random.randint(0, 7))

        plan_map[s["id"]] = plan["name"]
        rows.append({
            "id":         str(uuid.uuid4()),
            "student_id": s["id"],
            "plan_id":    plan["id"],
            "start_date": start_date.isoformat(),
        })

    return pd.DataFrame(rows), plan_map


# ── Motor de probabilidade de inadimplência ──────────────────────────────────

def _late_probability(age: int, plan_name: str, tenure_months: int) -> float:
    """
    Calcula a probabilidade de atraso com base nas regras de viés definidas.
    Não altere estas regras sem sincronizar com o notebook de feature engineering.
    """
    if 18 <= age <= 23 and plan_name == "Premium":
        prob = 0.60   # jovens + plano caro → maior risco
    elif age > 35:
        prob = 0.15   # adultos maduros → menor risco
    elif 24 <= age <= 35:
        prob = 0.30   # faixa intermediária
    else:
        prob = 0.35   # jovens em outros planos

    # Tenure alta = histórico de pagamentos confiável → desconto de risco
    if tenure_months > 12:
        prob *= 0.50
    elif tenure_months > 6:
        prob *= 0.75

    return prob


# ── Construção individual de pagamento ───────────────────────────────────────

def _build_payment(student_id: str, amount: float,
                   due_date: date, late_prob: float) -> dict:
    base = {
        "id":         str(uuid.uuid4()),
        "student_id": student_id,
        "amount":     amount,
        "due_date":   due_date.isoformat(),
    }

    if random.random() < late_prob:
        # 30 % dos inadimplentes nunca pagaram (overdue)
        if random.random() < 0.30:
            return {**base, "paid_at": None, "status": "pending"}

        # 70 % pagaram com atraso de 1 a 45 dias
        days_late  = random.randint(1, 45)
        late_date  = due_date + timedelta(days=days_late)
        paid_at    = late_date if late_date <= REFERENCE_DATE else None
        status     = "paid" if paid_at else "pending"
        return {**base, "paid_at": paid_at.isoformat() if paid_at else None, "status": status}

    # Pagamento pontual — 0 a 5 dias antes do vencimento
    days_early = random.randint(0, 5)
    return {
        **base,
        "paid_at": (due_date - timedelta(days=days_early)).isoformat(),
        "status":  "paid",
    }


# ── Geração de pagamentos ────────────────────────────────────────────────────

def make_payments(
    students: pd.DataFrame,
    enrollments: pd.DataFrame,
    plan_map: dict[str, str],
) -> pd.DataFrame:
    enrollment_idx = enrollments.set_index("student_id")
    rows: list[dict] = []

    for _, s in students.iterrows():
        sid        = s["id"]
        plan_name  = plan_map[sid]
        plan       = _PLAN_BY_NAME[plan_name]
        start_date = date.fromisoformat(enrollment_idx.loc[sid, "start_date"])
        birth_date = date.fromisoformat(s["birth_date"])
        created_at = date.fromisoformat(s["created_at"])

        age           = (REFERENCE_DATE - birth_date).days // 365
        tenure_months = (REFERENCE_DATE - created_at).days // 30
        late_prob     = _late_probability(age, plan_name, tenure_months)

        # Histórico de 1 a 12 mensalidades por aluno
        n_months = random.randint(1, 12)
        for month in range(1, n_months + 1):
            # Incremento mensal sem drift: adiciona ~30 dias por ciclo
            due_date = start_date + timedelta(days=30 * month)
            if due_date > REFERENCE_DATE:
                break
            rows.append(_build_payment(sid, plan["price"], due_date, late_prob))

    return pd.DataFrame(rows)


# ── Relatório de validação dos vieses ────────────────────────────────────────

def _print_validation_report(
    students: pd.DataFrame,
    enrollments: pd.DataFrame,
    payments: pd.DataFrame,
) -> None:
    """
    Calcula a taxa de inadimplência por grupo para confirmar que os vieses
    foram injetados corretamente antes de usar o dataset em treinamento.
    """
    # Constrói view enriquecida por aluno
    enr = enrollments[["student_id", "plan_id"]].copy()
    plan_name_map = {p["id"]: p["name"] for p in PLANS}
    enr["plan_name"] = enr["plan_id"].map(plan_name_map)

    stu = students.copy()
    stu["birth_date"] = pd.to_datetime(stu["birth_date"])
    stu["created_at"] = pd.to_datetime(stu["created_at"])
    ref = pd.Timestamp(REFERENCE_DATE)
    stu["age"]           = ((ref - stu["birth_date"]).dt.days // 365).astype(int)
    stu["tenure_months"] = ((ref - stu["created_at"]).dt.days // 30).astype(int)

    profile = stu[["id", "age", "tenure_months"]].merge(
        enr[["student_id", "plan_name"]], left_on="id", right_on="student_id"
    )

    # Taxa de inadimplência por aluno: pending ou pago com atraso
    pay = payments.copy()
    pay["is_bad"] = (pay["status"] == "pending") | (
        (pay["status"] == "paid") & (pay["paid_at"] > pay["due_date"])
    )
    bad_rate = pay.groupby("student_id")["is_bad"].mean().rename("bad_rate")
    profile  = profile.merge(bad_rate, left_on="id", right_index=True, how="left").fillna({"bad_rate": 0})

    # Grupos do requisito
    g_young_premium = profile[(profile["age"] <= 23) & (profile["plan_name"] == "Premium")]
    g_senior        = profile[profile["age"] > 35]
    g_high_tenure   = profile[profile["tenure_months"] > 12]

    # Totais globais
    total   = len(payments)
    pending = int((payments["status"] == "pending").sum())
    paid_df = payments[payments["status"] == "paid"]
    late    = int((paid_df["paid_at"] > paid_df["due_date"]).sum())

    print("\n" + "=" * 63)
    print("  Relatorio de Validacao do Dataset")
    print("=" * 63)
    print(f"  Total de pagamentos  : {total:>6,}")
    print(f"  Pendentes (sem pagar): {pending:>6,}  ({pending / total * 100:.1f} %)")
    print(f"  Pagos com atraso     : {late:>6,}  ({late / total * 100:.1f} %)")
    print(f"  Taxa geral de risco  : {(pending + late) / total * 100:.1f} %")
    print()
    print("  Validacao dos Vieses Injetados")

    def _fmt(df: pd.DataFrame, label: str, expected: str) -> None:
        rate = df["bad_rate"].mean() if len(df) else float("nan")
        print(f"  {label:<36} {rate:>5.1%}  (esperado: {expected})")

    _fmt(g_young_premium, "Jovens 18-23 c/ Premium (n=%d)" % len(g_young_premium), "~60%")
    _fmt(g_senior,        "Acima de 35 anos        (n=%d)" % len(g_senior),        "~15%")
    _fmt(g_high_tenure,   "Tenure > 12 meses       (n=%d)" % len(g_high_tenure),   "baixo")
    print("=" * 63 + "\n")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Gerando dataset sintético de risco de crédito...\n")

    print("  [1/4] Alunos")
    students = make_students(NUM_STUDENTS)

    print("  [2/4] Matrículas")
    enrollments, plan_map = make_enrollments(students)

    print("  [3/4] Histórico de pagamentos")
    payments = make_payments(students, enrollments, plan_map)

    print("  [4/4] Exportando CSVs")
    pd.DataFrame(PLANS).to_csv(DATA_DIR / "plans.csv",       index=False)
    students.to_csv(            DATA_DIR / "students.csv",    index=False)
    enrollments.to_csv(         DATA_DIR / "enrollments.csv", index=False)
    payments.to_csv(            DATA_DIR / "payments.csv",    index=False)

    print(f"\n  Arquivos salvos em: {DATA_DIR}")
    print(f"    plans.csv        :  {len(PLANS)} planos")
    print(f"    students.csv     :  {len(students):>5,} alunos")
    print(f"    enrollments.csv  :  {len(enrollments):>5,} matriculas")
    print(f"    payments.csv     :  {len(payments):>5,} pagamentos")

    _print_validation_report(students, enrollments, payments)


if __name__ == "__main__":
    main()
