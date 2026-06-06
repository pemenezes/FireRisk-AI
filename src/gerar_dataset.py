import numpy as np
import pandas as pd
import os


def classificar_risco(row):
    """
    Classifica o risco de incêndio com base em uma pontuação.
    Quanto maior a pontuação, maior o risco.
    """

    score = 0

    # Temperatura
    if row["temperatura_media"] >= 35:
        score += 3
    elif row["temperatura_media"] >= 30:
        score += 2
    elif row["temperatura_media"] >= 26:
        score += 1

    # Umidade
    if row["umidade_relativa"] <= 25:
        score += 3
    elif row["umidade_relativa"] <= 40:
        score += 2
    elif row["umidade_relativa"] <= 55:
        score += 1

    # Precipitação
    if row["precipitacao_mm"] <= 2:
        score += 3
    elif row["precipitacao_mm"] <= 10:
        score += 2
    elif row["precipitacao_mm"] <= 25:
        score += 1

    # Dias sem chuva
    if row["dias_sem_chuva"] >= 20:
        score += 3
    elif row["dias_sem_chuva"] >= 10:
        score += 2
    elif row["dias_sem_chuva"] >= 5:
        score += 1

    # Vento
    if row["velocidade_vento"] >= 30:
        score += 2
    elif row["velocidade_vento"] >= 18:
        score += 1

    # Focos de calor nos últimos 7 dias
    if row["focos_calor_7d"] >= 10:
        score += 3
    elif row["focos_calor_7d"] >= 5:
        score += 2
    elif row["focos_calor_7d"] >= 1:
        score += 1

    # Distância até foco mais próximo
    if row["distancia_foco_km"] <= 5:
        score += 3
    elif row["distancia_foco_km"] <= 15:
        score += 2
    elif row["distancia_foco_km"] <= 30:
        score += 1

    # NDVI: valores menores podem indicar vegetação mais seca/degradada
    if row["ndvi"] <= 0.25:
        score += 2
    elif row["ndvi"] <= 0.45:
        score += 1

    # Histórico de incêndios
    if row["historico_incendios"] >= 8:
        score += 2
    elif row["historico_incendios"] >= 4:
        score += 1

    # Biomas com maior sensibilidade sazonal a queimadas
    if row["bioma"] in ["Cerrado", "Amazônia", "Pantanal"]:
        score += 1

    # Meses tipicamente mais secos em muitas regiões do Brasil
    if row["mes"] in [7, 8, 9, 10]:
        score += 1

    # Classificação final
    if score <= 5:
        return "baixo"
    elif score <= 10:
        return "medio"
    elif score <= 15:
        return "alto"
    else:
        return "critico"


def gerar_dataset(qtd_linhas=1500, seed=42):
    np.random.seed(seed)

    estados = [
        "SP", "RJ", "MG", "MT", "MS", "GO", "TO", "PA", "AM", "RO",
        "BA", "MA", "PI", "PR", "RS", "SC", "PE", "CE"
    ]

    biomas = [
        "Amazônia",
        "Cerrado",
        "Mata Atlântica",
        "Caatinga",
        "Pantanal",
        "Pampa"
    ]

    dados = []

    for _ in range(qtd_linhas):
        estado = np.random.choice(estados)
        bioma = np.random.choice(
            biomas,
            p=[0.22, 0.28, 0.20, 0.12, 0.10, 0.08]
        )

        mes = np.random.randint(1, 13)

        # Coordenadas aproximadas dentro do Brasil
        latitude = np.round(np.random.uniform(-33.0, 5.0), 6)
        longitude = np.round(np.random.uniform(-73.0, -34.0), 6)

        # Ajuste sazonal simples
        periodo_seco = mes in [7, 8, 9, 10]

        if periodo_seco:
            temperatura_media = np.random.normal(31, 4)
            umidade_relativa = np.random.normal(38, 14)
            precipitacao_mm = np.random.exponential(8)
            dias_sem_chuva = np.random.randint(5, 35)
        else:
            temperatura_media = np.random.normal(27, 4)
            umidade_relativa = np.random.normal(62, 16)
            precipitacao_mm = np.random.exponential(25)
            dias_sem_chuva = np.random.randint(0, 18)

        # Ajustes por bioma
        if bioma in ["Cerrado", "Caatinga", "Pantanal"]:
            temperatura_media += np.random.uniform(1, 4)
            umidade_relativa -= np.random.uniform(5, 15)
            dias_sem_chuva += np.random.randint(0, 8)

        if bioma == "Amazônia":
            umidade_relativa += np.random.uniform(5, 15)
            precipitacao_mm += np.random.uniform(5, 25)

        if bioma == "Pampa":
            temperatura_media -= np.random.uniform(1, 4)

        # Limites realistas
        temperatura_media = np.clip(temperatura_media, 15, 43)
        umidade_relativa = np.clip(umidade_relativa, 10, 95)
        precipitacao_mm = np.clip(precipitacao_mm, 0, 180)
        dias_sem_chuva = np.clip(dias_sem_chuva, 0, 45)

        velocidade_vento = np.random.normal(15, 8)
        velocidade_vento = np.clip(velocidade_vento, 0, 55)

        # Focos de calor: mais prováveis no período seco e em certos biomas
        base_focos = 1

        if periodo_seco:
            base_focos += 3

        if bioma in ["Cerrado", "Amazônia", "Pantanal", "Caatinga"]:
            base_focos += 2

        if umidade_relativa < 35:
            base_focos += 2

        if dias_sem_chuva > 15:
            base_focos += 2

        focos_calor_7d = np.random.poisson(base_focos)
        focos_calor_7d = int(np.clip(focos_calor_7d, 0, 25))

        if focos_calor_7d == 0:
            distancia_foco_km = np.random.uniform(40, 150)
        else:
            distancia_foco_km = np.random.exponential(20)
            distancia_foco_km = np.clip(distancia_foco_km, 1, 120)

        # NDVI entre 0 e 1
        # Período seco e pouca chuva tendem a reduzir o índice
        ndvi = np.random.normal(0.58, 0.18)

        if periodo_seco:
            ndvi -= np.random.uniform(0.05, 0.18)

        if precipitacao_mm < 5:
            ndvi -= np.random.uniform(0.05, 0.15)

        if bioma in ["Caatinga", "Cerrado"]:
            ndvi -= np.random.uniform(0.03, 0.12)

        ndvi = np.clip(ndvi, 0.05, 0.95)

        historico_incendios = np.random.poisson(3)

        if bioma in ["Cerrado", "Amazônia", "Pantanal"]:
            historico_incendios += np.random.randint(0, 5)

        if periodo_seco:
            historico_incendios += np.random.randint(0, 3)

        historico_incendios = int(np.clip(historico_incendios, 0, 15))

        linha = {
            "latitude": latitude,
            "longitude": longitude,
            "estado": estado,
            "bioma": bioma,
            "mes": mes,
            "temperatura_media": round(float(temperatura_media), 2),
            "umidade_relativa": round(float(umidade_relativa), 2),
            "precipitacao_mm": round(float(precipitacao_mm), 2),
            "velocidade_vento": round(float(velocidade_vento), 2),
            "dias_sem_chuva": int(dias_sem_chuva),
            "focos_calor_7d": focos_calor_7d,
            "distancia_foco_km": round(float(distancia_foco_km), 2),
            "ndvi": round(float(ndvi), 3),
            "historico_incendios": historico_incendios
        }

        linha["risco_incendio"] = classificar_risco(linha)

        dados.append(linha)

    df = pd.DataFrame(dados)

    return df


if __name__ == "__main__":
    df = gerar_dataset(qtd_linhas=1500)

    os.makedirs("data", exist_ok=True)

    caminho_arquivo = "data/firerisk_dataset.csv"
    df.to_csv(caminho_arquivo, index=False, encoding="utf-8")

    print("Dataset gerado com sucesso!")
    print(f"Arquivo salvo em: {caminho_arquivo}")
    print()
    print("Dimensões do dataset:")
    print(df.shape)
    print()
    print("Primeiras linhas:")
    print(df.head())
    print()
    print("Distribuição das classes:")
    print(df["risco_incendio"].value_counts())