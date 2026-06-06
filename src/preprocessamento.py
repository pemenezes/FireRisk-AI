import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = "data/firerisk_dataset.csv"
TARGET = "risco_incendio"

COLUNAS_ESPERADAS = [
    "latitude",
    "longitude",
    "estado",
    "bioma",
    "mes",
    "temperatura_media",
    "umidade_relativa",
    "precipitacao_mm",
    "velocidade_vento",
    "dias_sem_chuva",
    "focos_calor_7d",
    "distancia_foco_km",
    "ndvi",
    "historico_incendios",
    "risco_incendio"
]


def carregar_dataset(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho)
    return df


def validar_colunas(df):
    colunas_atuais = list(df.columns)

    colunas_faltantes = [
        coluna for coluna in COLUNAS_ESPERADAS
        if coluna not in colunas_atuais
    ]

    colunas_extras = [
        coluna for coluna in colunas_atuais
        if coluna not in COLUNAS_ESPERADAS
    ]

    if colunas_faltantes:
        raise ValueError(f"Colunas faltantes: {colunas_faltantes}")

    if colunas_extras:
        print(f"Atenção: colunas extras encontradas: {colunas_extras}")

    print("Validação de colunas concluída com sucesso.")


def analisar_dataset(df):
    print("\nDimensões do dataset:")
    print(df.shape)

    print("\nTipos das colunas:")
    print(df.dtypes)

    print("\nQuantidade de valores nulos por coluna:")
    print(df.isnull().sum())

    print("\nQuantidade de linhas duplicadas:")
    print(df.duplicated().sum())

    print("\nDistribuição da variável alvo:")
    print(df[TARGET].value_counts())

    print("\nDistribuição percentual da variável alvo:")
    print((df[TARGET].value_counts(normalize=True) * 100).round(2))


def gerar_grafico_distribuicao_classes(df):
    os.makedirs("outputs/figures", exist_ok=True)

    ordem_classes = ["baixo", "medio", "alto", "critico"]
    contagem = df[TARGET].value_counts().reindex(ordem_classes)

    plt.figure(figsize=(8, 5))
    plt.bar(contagem.index, contagem.values)
    plt.title("Distribuição das Classes de Risco de Incêndio")
    plt.xlabel("Classe de risco")
    plt.ylabel("Quantidade de registros")
    plt.tight_layout()

    caminho_saida = "outputs/figures/distribuicao_classes.png"
    plt.savefig(caminho_saida)
    plt.close()

    print(f"\nGráfico salvo em: {caminho_saida}")


def salvar_resumo_dataset(df):
    os.makedirs("outputs", exist_ok=True)

    caminho_saida = "outputs/resumo_dataset.txt"

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Resumo do Dataset - FireRisk AI\n")
        arquivo.write("=" * 40)
        arquivo.write("\n\n")

        arquivo.write(f"Dimensões do dataset: {df.shape}\n\n")

        arquivo.write("Colunas:\n")
        for coluna in df.columns:
            arquivo.write(f"- {coluna}: {df[coluna].dtype}\n")

        arquivo.write("\nValores nulos por coluna:\n")
        arquivo.write(str(df.isnull().sum()))

        arquivo.write("\n\nLinhas duplicadas:\n")
        arquivo.write(str(df.duplicated().sum()))

        arquivo.write("\n\nDistribuição da variável alvo:\n")
        arquivo.write(str(df[TARGET].value_counts()))

        arquivo.write("\n\nDistribuição percentual da variável alvo:\n")
        arquivo.write(str((df[TARGET].value_counts(normalize=True) * 100).round(2)))

    print(f"Resumo salvo em: {caminho_saida}")


def separar_treino_teste(df):
    os.makedirs("data", exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[TARGET]
    )

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("\nSeparação treino/teste concluída.")
    print(f"Treino salvo em: {train_path}")
    print(f"Teste salvo em: {test_path}")
    print(f"Formato treino: {train_df.shape}")
    print(f"Formato teste: {test_df.shape}")

    print("\nDistribuição no treino:")
    print(train_df[TARGET].value_counts(normalize=True).round(3))

    print("\nDistribuição no teste:")
    print(test_df[TARGET].value_counts(normalize=True).round(3))


def main():
    df = carregar_dataset(DATA_PATH)

    validar_colunas(df)
    analisar_dataset(df)
    gerar_grafico_distribuicao_classes(df)
    salvar_resumo_dataset(df)
    separar_treino_teste(df)

    print("\nEtapa 4 concluída com sucesso!")


if __name__ == "__main__":
    main()