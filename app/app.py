import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/melhor_modelo.pkl"
SHAP_PATH = "outputs/shap/importancia_global_shap.csv"


ESTADOS = [
    "SP", "RJ", "MG", "MT", "MS", "GO", "TO", "PA", "AM", "RO",
    "BA", "MA", "PI", "PR", "RS", "SC", "PE", "CE"
]

BIOMAS = [
    "Amazônia",
    "Cerrado",
    "Mata Atlântica",
    "Caatinga",
    "Pantanal",
    "Pampa"
]


CENARIO_BAIXO_RISCO = {
    "estado": "SC",
    "bioma": "Mata Atlântica",
    "latitude": -27.60,
    "longitude": -48.55,
    "mes": 4,
    "temperatura_media": 22.0,
    "umidade_relativa": 78.0,
    "precipitacao_mm": 60.0,
    "velocidade_vento": 8.0,
    "dias_sem_chuva": 1,
    "focos_calor_7d": 0,
    "distancia_foco_km": 100.0,
    "ndvi": 0.75,
    "historico_incendios": 1
}


CENARIO_CRITICO = {
    "estado": "MT",
    "bioma": "Cerrado",
    "latitude": -15.60,
    "longitude": -56.10,
    "mes": 9,
    "temperatura_media": 38.0,
    "umidade_relativa": 18.0,
    "precipitacao_mm": 0.0,
    "velocidade_vento": 25.0,
    "dias_sem_chuva": 25,
    "focos_calor_7d": 12,
    "distancia_foco_km": 3.0,
    "ndvi": 0.22,
    "historico_incendios": 9
}


def configurar_pagina():
    st.set_page_config(
        page_title="FireRisk AI",
        page_icon="🔥",
        layout="wide"
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 46px;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 22px;
            color: #374151;
            margin-bottom: 20px;
        }

        .risk-card {
            padding: 24px;
            border-radius: 14px;
            margin-top: 10px;
            margin-bottom: 18px;
            font-size: 24px;
            font-weight: 700;
            text-align: center;
        }

        .risk-baixo {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }

        .risk-medio {
            background-color: #dbeafe;
            color: #1e40af;
            border: 1px solid #93c5fd;
        }

        .risk-alto {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
        }

        .risk-critico {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }

        .info-box {
            padding: 18px;
            border-radius: 12px;
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            margin-bottom: 16px;
        }

        .metric-box {
            padding: 16px;
            border-radius: 12px;
            background-color: #f3f4f6;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def inicializar_estado():
    for chave, valor in CENARIO_CRITICO.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


def aplicar_cenario_baixo():
    for chave, valor in CENARIO_BAIXO_RISCO.items():
        st.session_state[chave] = valor


def aplicar_cenario_critico():
    for chave, valor in CENARIO_CRITICO.items():
        st.session_state[chave] = valor


def carregar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.error("Modelo não encontrado. Execute primeiro o script de treinamento.")
        st.stop()

    return joblib.load(MODEL_PATH)


def carregar_shap():
    if os.path.exists(SHAP_PATH):
        return pd.read_csv(SHAP_PATH)

    return None


def gerar_recomendacoes(risco):
    recomendacoes = {
        "baixo": [
            "Manter monitoramento periódico da região.",
            "Registrar histórico climático para análises futuras.",
            "Não há necessidade de ação emergencial neste momento."
        ],
        "medio": [
            "Acompanhar variações de umidade e temperatura.",
            "Evitar queimadas controladas sem autorização.",
            "Monitorar focos de calor próximos nos próximos dias."
        ],
        "alto": [
            "Emitir alerta preventivo para equipes locais.",
            "Intensificar fiscalização em áreas rurais e de vegetação seca.",
            "Preparar equipes de resposta para possível ocorrência."
        ],
        "critico": [
            "Acionar protocolo de alerta imediato.",
            "Priorizar monitoramento da região em tempo reduzido.",
            "Mobilizar equipes de prevenção e resposta.",
            "Evitar qualquer atividade com risco de ignição na área."
        ]
    }

    return recomendacoes.get(risco, ["Sem recomendações disponíveis."])


def gerar_explicacao_resultado(risco, entrada_df):
    dados = entrada_df.iloc[0]

    fatores = []

    if dados["umidade_relativa"] <= 30:
        fatores.append("umidade relativa muito baixa")

    if dados["temperatura_media"] >= 35:
        fatores.append("temperatura elevada")

    if dados["precipitacao_mm"] <= 5:
        fatores.append("baixa precipitação acumulada")

    if dados["dias_sem_chuva"] >= 15:
        fatores.append("muitos dias consecutivos sem chuva")

    if dados["focos_calor_7d"] >= 5:
        fatores.append("presença de focos de calor recentes")

    if dados["distancia_foco_km"] <= 10:
        fatores.append("proximidade com focos de calor")

    if dados["ndvi"] <= 0.30:
        fatores.append("baixo índice de vegetação, indicando possível vegetação seca ou degradada")

    if dados["historico_incendios"] >= 6:
        fatores.append("histórico relevante de incêndios na região")

    if not fatores:
        fatores.append("condições ambientais pouco favoráveis à ocorrência de incêndios")

    texto_fatores = ", ".join(fatores)

    return (
        f"O modelo classificou a região como risco **{risco.upper()}**. "
        f"Os principais fatores observados foram: {texto_fatores}."
    )


def montar_dataframe_entrada():
    dados = {
        "latitude": [st.session_state["latitude"]],
        "longitude": [st.session_state["longitude"]],
        "estado": [st.session_state["estado"]],
        "bioma": [st.session_state["bioma"]],
        "mes": [st.session_state["mes"]],
        "temperatura_media": [st.session_state["temperatura_media"]],
        "umidade_relativa": [st.session_state["umidade_relativa"]],
        "precipitacao_mm": [st.session_state["precipitacao_mm"]],
        "velocidade_vento": [st.session_state["velocidade_vento"]],
        "dias_sem_chuva": [st.session_state["dias_sem_chuva"]],
        "focos_calor_7d": [st.session_state["focos_calor_7d"]],
        "distancia_foco_km": [st.session_state["distancia_foco_km"]],
        "ndvi": [st.session_state["ndvi"]],
        "historico_incendios": [st.session_state["historico_incendios"]]
    }

    return pd.DataFrame(dados)


def exibir_card_risco(risco):
    classe_css = {
        "baixo": "risk-baixo",
        "medio": "risk-medio",
        "alto": "risk-alto",
        "critico": "risk-critico"
    }.get(risco, "risk-medio")

    st.markdown(
        f"""
        <div class="risk-card {classe_css}">
            Risco previsto: {risco.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )


def exibir_sidebar():
    st.sidebar.title("🔥 FireRisk AI")
    st.sidebar.write("Demonstração rápida do MVP.")

    st.sidebar.divider()

    st.sidebar.subheader("Cenários prontos")

    st.sidebar.button(
        "Carregar cenário de baixo risco",
        on_click=aplicar_cenario_baixo,
        use_container_width=True
    )

    st.sidebar.button(
        "Carregar cenário crítico",
        on_click=aplicar_cenario_critico,
        use_container_width=True
    )

    st.sidebar.divider()

    st.sidebar.subheader("Sobre o MVP")
    st.sidebar.write(
        "Esta versão utiliza um dataset sintético realista para validar "
        "o pipeline completo de IA, incluindo treinamento, validação, SHAP e deploy."
    )


def exibir_cabecalho():
    st.markdown('<div class="main-title">🔥 FireRisk AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Sistema Inteligente de Previsão de Risco de Incêndios com Dados Orbitais</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
        O FireRisk AI estima o risco de incêndio em uma região com base em variáveis
        ambientais, climáticas, históricas e orbitais simuladas. O modelo foi treinado
        para classificar o risco em quatro níveis: <b>baixo</b>, <b>médio</b>,
        <b>alto</b> e <b>crítico</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


def exibir_formulario():
    st.header("Entrada de dados da região")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📍 Localização")

        st.selectbox("Estado", ESTADOS, key="estado")
        st.selectbox("Bioma", BIOMAS, key="bioma")

        st.number_input(
            "Latitude",
            min_value=-33.0,
            max_value=5.0,
            step=0.01,
            key="latitude"
        )

        st.number_input(
            "Longitude",
            min_value=-73.0,
            max_value=-34.0,
            step=0.01,
            key="longitude"
        )

        st.slider("Mês da análise", 1, 12, key="mes")

    with col2:
        st.subheader("🌦️ Condições ambientais")

        st.slider(
            "Temperatura média (°C)",
            min_value=15.0,
            max_value=43.0,
            step=0.1,
            key="temperatura_media"
        )

        st.slider(
            "Umidade relativa (%)",
            min_value=10.0,
            max_value=95.0,
            step=0.1,
            key="umidade_relativa"
        )

        st.slider(
            "Precipitação acumulada (mm)",
            min_value=0.0,
            max_value=180.0,
            step=0.5,
            key="precipitacao_mm"
        )

        st.slider(
            "Velocidade do vento (km/h)",
            min_value=0.0,
            max_value=55.0,
            step=0.5,
            key="velocidade_vento"
        )

    st.subheader("🔥 Indicadores de risco")

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.number_input(
            "Dias sem chuva",
            min_value=0,
            max_value=45,
            key="dias_sem_chuva"
        )

    with col4:
        st.number_input(
            "Focos de calor nos últimos 7 dias",
            min_value=0,
            max_value=25,
            key="focos_calor_7d"
        )

    with col5:
        st.number_input(
            "Distância do foco mais próximo (km)",
            min_value=1.0,
            max_value=150.0,
            step=0.5,
            key="distancia_foco_km"
        )

    with col6:
        st.number_input(
            "Histórico de incêndios",
            min_value=0,
            max_value=15,
            key="historico_incendios"
        )

    st.slider(
        "NDVI - Índice de vegetação",
        min_value=0.05,
        max_value=0.95,
        step=0.01,
        key="ndvi"
    )


def exibir_resultado(modelo, entrada_df):
    predicao = modelo.predict(entrada_df)[0]
    probabilidades = modelo.predict_proba(entrada_df)[0]
    classes = modelo.classes_

    st.header("📊 Resultado da análise")

    exibir_card_risco(predicao)

    st.markdown(gerar_explicacao_resultado(predicao, entrada_df))

    prob_df = pd.DataFrame({
        "Classe de risco": classes,
        "Probabilidade": probabilidades
    })

    prob_df["Probabilidade (%)"] = (prob_df["Probabilidade"] * 100).round(2)
    prob_df = prob_df.sort_values(by="Probabilidade (%)", ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Probabilidade por classe")
        st.dataframe(
            prob_df[["Classe de risco", "Probabilidade (%)"]],
            use_container_width=True
        )

    with col2:
        st.subheader("Distribuição das probabilidades")
        st.bar_chart(
            prob_df.set_index("Classe de risco")["Probabilidade"]
        )

    st.subheader("🧭 Recomendações preventivas")

    for recomendacao in gerar_recomendacoes(predicao):
        st.write(f"- {recomendacao}")

    with st.expander("Ver dados utilizados na previsão"):
        st.dataframe(entrada_df, use_container_width=True)


def exibir_shap(shap_df):
    st.header("🔎 Interpretabilidade do modelo com SHAP")

    if shap_df is None:
        st.warning(
            "Arquivo de SHAP não encontrado. Execute primeiro o script src/explicar_shap.py."
        )
        return

    st.write(
        "As variáveis abaixo foram identificadas como as mais influentes nas decisões "
        "do modelo, considerando a análise global com SHAP."
    )

    top_shap = shap_df.head(10).copy()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.dataframe(top_shap, use_container_width=True)

    with col2:
        grafico_df = top_shap.set_index("variavel")["importancia_media_abs_shap"]
        st.bar_chart(grafico_df)


def main():
    configurar_pagina()
    inicializar_estado()

    modelo = carregar_modelo()
    shap_df = carregar_shap()

    exibir_sidebar()
    exibir_cabecalho()

    st.divider()

    exibir_formulario()

    st.divider()

    analisar = st.button("Analisar risco de incêndio", type="primary")

    if analisar:
        entrada_df = montar_dataframe_entrada()
        exibir_resultado(modelo, entrada_df)

    st.divider()

    exibir_shap(shap_df)

    st.divider()

    st.caption(
        "FireRisk AI — MVP acadêmico com dataset sintético realista. "
        "Em uma evolução futura, a solução poderá ser integrada a APIs reais, "
        "como NASA FIRMS, INPE Queimadas e NASA POWER."
    )


if __name__ == "__main__":
    main()