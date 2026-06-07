# ML Service — Risco de Inadimplência

Microsserviço Python (FastAPI) responsável pela predição de risco de inadimplência de alunos de academia. Atua exclusivamente como **camada de inferência**: recebe um payload já processado pelo backend Java, aplica o modelo treinado e devolve a probabilidade de inadimplência por aluno.

---

## Estrutura do Repositório

```
ml-service/
├── api/
│   ├── routes.py          # POST /predict — lógica de inferência
│   └── schemas.py         # Modelos Pydantic de entrada e saída
├── data/
│   └── mock/              # CSVs sintéticos gerados por scripts/generate_data.py
│       ├── students.csv
│       ├── plans.csv
│       ├── enrollments.csv
│       └── payments.csv
├── docs/
│   ├── ML_SERVICE_SPEC.md # Contrato completo da API (request/response)
│   ├── banco-de-dados.md  # Schema do banco de dados relacional
│   └── fluxo.md           # Fluxo de comunicação Frontend → Java → Python
├── models/
│   └── default_risk_model.pkl  # Pipeline scikit-learn serializado (gerado localmente)
├── scripts/
│   ├── generate_data.py   # Gera dataset sintético em data/mock/
│   └── train_model.py     # Treina o modelo e salva em models/
├── tests/
│   └── test_predict.py    # Testes do endpoint /predict
├── tools/
│   └── rtp.ps1            # Script PowerShell para executar scripts Python
├── main.py                # Entrypoint FastAPI com carregamento do modelo
├── requirements.txt
└── .gitignore
```

> `data/` e `models/` estão no `.gitignore` — são artefatos gerados localmente pelos scripts.

---

## Fluxo de Dados

```
Frontend (React)
    │  clique em "Analisar Risco"
    ▼
Backend Java
    │  SELECT + JOIN + agregações no PostgreSQL
    │  POST /predict  →  FastAPI
    ▼
ML Service (este repositório)
    │  valida payload (Pydantic)
    │  monta array NumPy [n_alunos × 7 features]
    │  model.predict_proba(X)
    │  aplica threshold ≥ 0.70 → is_high_risk
    ▼
Backend Java
    │  persiste em risk_assessments (score, category, created_at)
    ▼
Frontend (React)
    └  exibe relatório de risco por aluno
```

---

## Setup

**Pré-requisito:** Python 3.11+

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Executando o Pipeline

### 1. Gerar dados sintéticos

```bash
python scripts/generate_data.py
```

Cria os 4 CSVs em `data/mock/` com 2.000 alunos simulados e vieses de risco injetados.

### 2. Treinar o modelo

```bash
python scripts/train_model.py
```

Processa os CSVs, calcula as 7 features, treina um `RandomForestClassifier` dentro de um `Pipeline` com `StandardScaler` e salva o resultado em `models/default_risk_model.pkl`.

Saída esperada no terminal:

```
AUC-ROC : 0.9973
accuracy : 0.97
```

### 3. Subir a API

```bash
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`. Documentação interativa: `http://localhost:8000/docs`.

---

## API

### `POST /predict`

**Request:**

```json
{
  "students": [
    {
      "student_id": "d9a0bf05-ad6d-47cc-a06a-c2230d4c0d46",
      "idade": 22,
      "preco_plano": 200.0,
      "meses_duracao_plano": 12,
      "meses_desde_matricula": 3,
      "media_dias_atraso": 14.5,
      "historico_faturas_atrasadas": 2,
      "total_faturas_pagas": 3
    }
  ]
}
```

| Campo | Tipo | Descrição |
| :--- | :---: | :--- |
| `student_id` | UUID | Identificador do aluno — retornado na resposta para correlação. Não entra no vetor de features. |
| `idade` | int | Idade em anos inteiros. |
| `preco_plano` | float | Valor da mensalidade. |
| `meses_duracao_plano` | int | Duração total do contrato em meses. |
| `meses_desde_matricula` | int | Tempo de relacionamento em meses. |
| `media_dias_atraso` | float | Média de dias de atraso no histórico. `0.0` se sem histórico. |
| `historico_faturas_atrasadas` | int | Contagem de faturas atrasadas ou em aberto. |
| `total_faturas_pagas` | int | Total de faturas liquidadas. |

**Response:**

```json
{
  "predictions": [
    {
      "student_id": "d9a0bf05-ad6d-47cc-a06a-c2230d4c0d46",
      "default_probability": 0.74,
      "is_high_risk": true
    }
  ]
}
```

| Campo | Tipo | Descrição |
| :--- | :---: | :--- |
| `default_probability` | float | Probabilidade de inadimplência entre `0.0` e `1.0`. |
| `is_high_risk` | bool | `true` quando `default_probability >= 0.70`. |

---

## Modelo

| Atributo | Valor |
| :--- | :--- |
| Algoritmo | Random Forest (`scikit-learn`) |
| Pré-processamento | `StandardScaler` embutido no `Pipeline` |
| Serialização | `joblib` → `models/default_risk_model.pkl` |
| Threshold de risco alto | `>= 0.70` |
| AUC-ROC (dataset sintético) | 0.9973 |

O modelo é carregado uma única vez no startup da aplicação (via `lifespan` do FastAPI) e mantido em `app.state.model`, sem leitura de disco por requisição.

---