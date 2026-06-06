import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


TARGET = "risco_incendio"

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
MELHOR_MODELO_PATH = "models/melhor_modelo.pkl"

OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
SHAP_DIR = "outputs/shap"


def limpar_nome_feature(nome):
    return (
        nome.replace("num__", "")
        .replace("cat__", "")
        .replace("estado_", "estado=")
        .replace("bioma_", "bioma=")
    )


def carregar_dados_e_modelo():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_PATH}")

    if not os.path.exists(MELHOR_MODELO_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {MELHOR_MODELO_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    modelo_pipeline = joblib.load(MELHOR_MODELO_PATH)

    return train_df, test_df, modelo_pipeline


def preparar_dados_transformados(train_df, test_df, modelo_pipeline):
    X_train = train_df.drop(columns=[TARGET])
    X_test = test_df.drop(columns=[TARGET])

    preprocessador = modelo_pipeline.named_steps["preprocessador"]
    modelo = modelo_pipeline.named_steps["modelo"]

    X_train_transformado = preprocessador.transform(X_train)
    X_test_transformado = preprocessador.transform(X_test)

    nomes_features = preprocessador.get_feature_names_out()
    nomes_features = [limpar_nome_feature(nome) for nome in nomes_features]

    return X_train, X_test, X_train_transformado, X_test_transformado, modelo, nomes_features


def calcular_shap(X_train_transformado, X_test_transformado, modelo, nomes_features):
    """
    Usa SHAP de forma compatível com modelos multiclasse.
    Para deixar a execução leve, usamos uma amostra do treino como background
    e uma amostra do teste para explicação.
    """

    np.random.seed(42)

    qtd_background = min(80, X_train_transformado.shape[0])
    qtd_explicacao = min(120, X_test_transformado.shape[0])

    indices_background = np.random.choice(
        X_train_transformado.shape[0],
        qtd_background,
        replace=False
    )

    indices_explicacao = np.random.choice(
        X_test_transformado.shape[0],
        qtd_explicacao,
        replace=False
    )

    background = X_train_transformado[indices_background]
    X_explicacao = X_test_transformado[indices_explicacao]

    print("Calculando valores SHAP...")
    print("Esta etapa pode demorar um pouco.")

    explainer = shap.PermutationExplainer(
        modelo.predict_proba,
        background,
        feature_names=nomes_features
    )

    max_evals = 2 * X_explicacao.shape[1] + 1

    shap_values = explainer(
        X_explicacao,
        max_evals=max_evals
    )

    return shap_values, X_explicacao, indices_explicacao


def obter_importancia_global(shap_values, nomes_features):
    valores = shap_values.values

    if valores.ndim == 3:
        if valores.shape[1] == len(nomes_features):
            importancia = np.abs(valores).mean(axis=(0, 2))
        elif valores.shape[2] == len(nomes_features):
            importancia = np.abs(valores).mean(axis=(0, 1))
        else:
            raise ValueError("Formato inesperado dos valores SHAP.")
    elif valores.ndim == 2:
        importancia = np.abs(valores).mean(axis=0)
    else:
        raise ValueError("Formato inesperado dos valores SHAP.")

    importancia_df = pd.DataFrame({
        "variavel": nomes_features,
        "importancia_media_abs_shap": importancia
    })

    importancia_df = importancia_df.sort_values(
        by="importancia_media_abs_shap",
        ascending=False
    )

    return importancia_df


def salvar_importancia_global(importancia_df):
    os.makedirs(SHAP_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    caminho_csv = f"{SHAP_DIR}/importancia_global_shap.csv"
    importancia_df.to_csv(caminho_csv, index=False, encoding="utf-8")

    top_n = 15
    top_features = importancia_df.head(top_n).sort_values(
        by="importancia_media_abs_shap",
        ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["variavel"],
        top_features["importancia_media_abs_shap"]
    )
    plt.title("Top 15 Variáveis Mais Influentes - SHAP")
    plt.xlabel("Importância média absoluta SHAP")
    plt.ylabel("Variável")
    plt.tight_layout()

    caminho_figura = f"{FIGURES_DIR}/shap_importancia_global.png"
    plt.savefig(caminho_figura)
    plt.close()

    print(f"Importância global salva em: {caminho_csv}")
    print(f"Gráfico SHAP global salvo em: {caminho_figura}")


def obter_valores_locais(shap_values, indice_amostra, indice_classe, nomes_features):
    valores = shap_values.values

    if valores.ndim == 3:
        if valores.shape[1] == len(nomes_features):
            valores_locais = valores[indice_amostra, :, indice_classe]
        elif valores.shape[2] == len(nomes_features):
            valores_locais = valores[indice_amostra, indice_classe, :]
        else:
            raise ValueError("Formato inesperado dos valores SHAP.")
    elif valores.ndim == 2:
        valores_locais = valores[indice_amostra, :]
    else:
        raise ValueError("Formato inesperado dos valores SHAP.")

    return valores_locais


def salvar_explicacao_local(shap_values, X_explicacao, modelo, nomes_features):
    os.makedirs(SHAP_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    probabilidades = modelo.predict_proba(X_explicacao)
    classes = list(modelo.classes_)
    predicoes = modelo.predict(X_explicacao)

    if "critico" in predicoes:
        indice_amostra = list(predicoes).index("critico")
    elif "alto" in predicoes:
        indice_amostra = list(predicoes).index("alto")
    else:
        indice_amostra = int(np.argmax(probabilidades.max(axis=1)))

    classe_prevista = predicoes[indice_amostra]
    indice_classe = classes.index(classe_prevista)

    valores_locais = obter_valores_locais(
        shap_values,
        indice_amostra,
        indice_classe,
        nomes_features
    )

    explicacao_df = pd.DataFrame({
        "variavel": nomes_features,
        "valor_shap": valores_locais,
        "impacto_absoluto": np.abs(valores_locais)
    })

    explicacao_df = explicacao_df.sort_values(
        by="impacto_absoluto",
        ascending=False
    )

    caminho_csv = f"{SHAP_DIR}/explicacao_local_exemplo.csv"
    explicacao_df.to_csv(caminho_csv, index=False, encoding="utf-8")

    top_n = 10
    top_local = explicacao_df.head(top_n).sort_values(
        by="impacto_absoluto",
        ascending=True
    )

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_local["variavel"],
        top_local["valor_shap"]
    )
    plt.title(f"Explicação Local SHAP - Previsão: {classe_prevista}")
    plt.xlabel("Valor SHAP")
    plt.ylabel("Variável")
    plt.tight_layout()

    caminho_figura = f"{FIGURES_DIR}/shap_explicacao_local.png"
    plt.savefig(caminho_figura)
    plt.close()

    print(f"Explicação local salva em: {caminho_csv}")
    print(f"Gráfico SHAP local salvo em: {caminho_figura}")

    return classe_prevista, explicacao_df


def salvar_relatorio_shap(importancia_df, classe_prevista, explicacao_local_df):
    caminho_saida = f"{OUTPUTS_DIR}/relatorio_shap.txt"

    top_global = importancia_df.head(10)
    top_local = explicacao_local_df.head(10)

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Relatório de Interpretabilidade SHAP - FireRisk AI\n")
        arquivo.write("=" * 60)
        arquivo.write("\n\n")

        arquivo.write("1. Objetivo\n")
        arquivo.write("-" * 60)
        arquivo.write("\n")
        arquivo.write(
            "Esta etapa utiliza SHAP para explicar quais variáveis mais influenciam "
            "as previsões do modelo de risco de incêndio. A análise ajuda a tornar "
            "o modelo mais transparente e interpretável para usuários finais, como "
            "Defesa Civil, prefeituras, brigadas ambientais e produtores rurais.\n\n"
        )

        arquivo.write("2. Variáveis mais influentes globalmente\n")
        arquivo.write("-" * 60)
        arquivo.write("\n")
        arquivo.write(top_global.to_string(index=False))
        arquivo.write("\n\n")

        arquivo.write("3. Exemplo de explicação local\n")
        arquivo.write("-" * 60)
        arquivo.write("\n")
        arquivo.write(f"Classe prevista no exemplo analisado: {classe_prevista}\n\n")
        arquivo.write(top_local.to_string(index=False))
        arquivo.write("\n\n")

        arquivo.write("4. Interpretação técnica\n")
        arquivo.write("-" * 60)
        arquivo.write("\n")
        arquivo.write(
            "As variáveis com maior importância média absoluta SHAP são aquelas que mais "
            "contribuíram para as decisões do modelo no conjunto analisado. Em um cenário "
            "de risco de incêndio, espera-se que fatores como umidade relativa, temperatura, "
            "precipitação, dias sem chuva, focos de calor próximos, distância até focos e NDVI "
            "tenham forte influência nas previsões. A explicação local demonstra, para uma "
            "previsão específica, quais fatores empurraram a decisão do modelo para determinada "
            "classe de risco."
        )

    print(f"Relatório SHAP salvo em: {caminho_saida}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(SHAP_DIR, exist_ok=True)

    train_df, test_df, modelo_pipeline = carregar_dados_e_modelo()

    (
        X_train,
        X_test,
        X_train_transformado,
        X_test_transformado,
        modelo,
        nomes_features
    ) = preparar_dados_transformados(train_df, test_df, modelo_pipeline)

    print("Dados e modelo carregados com sucesso.")
    print(f"Modelo utilizado: {type(modelo).__name__}")
    print(f"Quantidade de variáveis após pré-processamento: {len(nomes_features)}")

    shap_values, X_explicacao, indices_explicacao = calcular_shap(
        X_train_transformado,
        X_test_transformado,
        modelo,
        nomes_features
    )

    importancia_df = obter_importancia_global(shap_values, nomes_features)
    salvar_importancia_global(importancia_df)

    classe_prevista, explicacao_local_df = salvar_explicacao_local(
        shap_values,
        X_explicacao,
        modelo,
        nomes_features
    )

    salvar_relatorio_shap(
        importancia_df,
        classe_prevista,
        explicacao_local_df
    )

    print("\nTop 10 variáveis mais importantes segundo SHAP:")
    print(importancia_df.head(10))

    print("\nEtapa 7 concluída com sucesso!")


if __name__ == "__main__":
    main()