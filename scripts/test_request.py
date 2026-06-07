import urllib.request
import urllib.error
import json

API_URL = "http://localhost:8000/predict"

STUDENTS = [
    {
        "student_id": "a3f1c2d4-e5b6-4789-8abc-100000000001",
        "nome":       "Carlos, 19 anos, plano Premium, 1 mes, 3 atrasos",
        "idade": 19,
        "preco_plano": 200.0,
        "meses_duracao_plano": 12,
        "meses_desde_matricula": 1,
        "media_dias_atraso": 22.0,
        "historico_faturas_atrasadas": 3,
        "total_faturas_pagas": 1,
    },
    {
        "student_id": "b4e2d3f5-f6c7-4890-9bcd-200000000002",
        "nome":       "Fernanda, 45 anos, plano Basico, 36 meses, sem atrasos",
        "idade": 45,
        "preco_plano": 50.0,
        "meses_duracao_plano": 12,
        "meses_desde_matricula": 36,
        "media_dias_atraso": 0.0,
        "historico_faturas_atrasadas": 0,
        "total_faturas_pagas": 36,
    },
    {
        "student_id": "c5f3e4a6-a7d8-4901-aced-300000000003",
        "nome":       "Rafael, 28 anos, plano Padrao, 6 meses, 1 atraso leve",
        "idade": 28,
        "preco_plano": 100.0,
        "meses_duracao_plano": 12,
        "meses_desde_matricula": 6,
        "media_dias_atraso": 5.0,
        "historico_faturas_atrasadas": 1,
        "total_faturas_pagas": 5,
    },
    {
        "student_id": "d6a4f5b7-b8e9-4a12-bdef-400000000004",
        "nome":       "Julia, 21 anos, plano Premium, 2 meses, inadimplente recorrente",
        "idade": 21,
        "preco_plano": 200.0,
        "meses_duracao_plano": 12,
        "meses_desde_matricula": 2,
        "media_dias_atraso": 38.0,
        "historico_faturas_atrasadas": 5,
        "total_faturas_pagas": 0,
    },
    {
        "student_id": "e7b5a6c8-c9fa-4b23-aeaf-500000000005",
        "nome":       "Marcos, 55 anos, plano Basico, 60 meses, pagador pontual",
        "idade": 55,
        "preco_plano": 50.0,
        "meses_duracao_plano": 12,
        "meses_desde_matricula": 60,
        "media_dias_atraso": 0.0,
        "historico_faturas_atrasadas": 0,
        "total_faturas_pagas": 60,
    },
]

FEATURE_KEYS = [
    "student_id", "idade", "preco_plano", "meses_duracao_plano",
    "meses_desde_matricula", "media_dias_atraso",
    "historico_faturas_atrasadas", "total_faturas_pagas",
]

id_to_nome = {s["student_id"]: s["nome"] for s in STUDENTS}
payload = {"students": [{k: s[k] for k in FEATURE_KEYS} for s in STUDENTS]}


def main():
    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"Erro ao conectar na API: {e.reason}")
        print("Verifique se a API esta rodando: uvicorn main:app --reload")
        return

    print(f"\n{'='*62}")
    print(f"  Resultado: {len(result['predictions'])} predicoes recebidas")
    print(f"{'='*62}")

    for pred in result["predictions"]:
        sid   = pred["student_id"]
        nome  = id_to_nome.get(sid, "desconhecido")
        prob  = pred["default_probability"]
        risco = "ALTO  [!]" if pred["is_high_risk"] else "baixo"

        print(f"\n  Aluno : {nome}")
        print(f"  ID    : {sid}")
        print(f"  Prob  : {prob:.2%}")
        print(f"  Risco : {risco}")

    print(f"\n{'='*62}")
    print("  A API retornou o student_id em cada predicao,")
    print("  permitindo correlacionar o resultado ao aluno de origem.")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
