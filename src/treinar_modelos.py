import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


TARGET = "risco_incendio"

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"

CLASSES = ["baixo", "medio", "alto", "critico"]


def criar_onehot_encoder():
    """
    Função para evitar erro entre versões diferentes do scikit-learn.
    Algumas versões usam sparse_output=False.
    Outras versões usam sparse=False.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def carregar_dados():
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TRAIN_PATH}")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test


def criar_preprocessador():
    colunas_numericas = [
        "latitude",
        "longitude",
        "mes",
        "temperatura_media",
        "umidade_relativa",
        "precipitacao_mm",
        "velocidade_vento",
        "dias_sem_chuva",
        "focos_calor_7d",
        "distancia_foco_km",
        "ndvi",
        "historico_incendios"
    ]

    colunas_categoricas = [
        "estado",
        "bioma"
    ]

    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", criar_onehot_encoder())
    ])

    preprocessador = ColumnTransformer(
        transformers=[
            ("num", transformador_numerico, colunas_numericas),
            ("cat", transformador_categorico, colunas_categoricas)
        ]
    )

    return preprocessador


def criar_modelos(preprocessador):
    random_forest = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("modelo", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    gradient_boosting = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("modelo", GradientBoostingClassifier(
            random_state=42
        ))
    ])

    modelos = {
        "Random Forest": random_forest,
        "Gradient Boosting": gradient_boosting
    }

    return modelos


def salvar_matriz_confusao(nome_modelo, y_test, y_pred):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    matriz = confusion_matrix(y_test, y_pred, labels=CLASSES)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=CLASSES
    )

    display.plot(cmap=None)
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.tight_layout()

    nome_arquivo = nome_modelo.lower().replace(" ", "_")
    caminho_saida = f"{FIGURES_DIR}/matriz_confusao_{nome_arquivo}.png"

    plt.savefig(caminho_saida)
    plt.close()

    print(f"Matriz de confusão salva em: {caminho_saida}")


def avaliar_modelo(nome_modelo, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"Resultados do modelo: {nome_modelo}")
    print("=" * 60)
    print(f"Acurácia: {acuracia:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro: {recall_macro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0
    ))

    salvar_matriz_confusao(nome_modelo, y_test, y_pred)

    resultado = {
        "modelo": nome_modelo,
        "acuracia": acuracia,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

    return resultado


def salvar_relatorio_texto(resultados_df, melhor_modelo_nome):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    caminho_saida = f"{OUTPUTS_DIR}/relatorio_treinamento.txt"

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Relatório de Treinamento - FireRisk AI\n")
        arquivo.write("=" * 50)
        arquivo.write("\n\n")

        arquivo.write("Modelos treinados:\n")
        arquivo.write("- Random Forest Classifier\n")
        arquivo.write("- Gradient Boosting Classifier\n\n")

        arquivo.write("Métricas comparativas:\n")
        arquivo.write(resultados_df.to_string(index=False))

        arquivo.write("\n\nMelhor modelo escolhido:\n")
        arquivo.write(melhor_modelo_nome)

        arquivo.write("\n\nCritério de escolha:\n")
        arquivo.write("Maior valor de F1 Macro, pois o dataset possui classes moderadamente desbalanceadas.")

    print(f"\nRelatório salvo em: {caminho_saida}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = carregar_dados()

    print("Dados carregados com sucesso.")
    print(f"Formato X_train: {X_train.shape}")
    print(f"Formato X_test: {X_test.shape}")
    print(f"Formato y_train: {y_train.shape}")
    print(f"Formato y_test: {y_test.shape}")

    preprocessador = criar_preprocessador()
    modelos = criar_modelos(preprocessador)

    resultados = {}
    modelos_treinados = {}

    pesos_treino = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    for nome_modelo, modelo in modelos.items():
        print("\n" + "#" * 60)
        print(f"Treinando modelo: {nome_modelo}")
        print("#" * 60)

        if nome_modelo == "Gradient Boosting":
            modelo.fit(X_train, y_train, modelo__sample_weight=pesos_treino)
        else:
            modelo.fit(X_train, y_train)

        modelos_treinados[nome_modelo] = modelo

        nome_arquivo_modelo = nome_modelo.lower().replace(" ", "_")
        caminho_modelo = f"{MODELS_DIR}/{nome_arquivo_modelo}_pipeline.pkl"

        joblib.dump(modelo, caminho_modelo)
        print(f"Modelo salvo em: {caminho_modelo}")

        resultado = avaliar_modelo(nome_modelo, modelo, X_test, y_test)
        resultados[nome_modelo] = resultado

    resultados_df = pd.DataFrame(resultados.values())
    resultados_df = resultados_df.sort_values(by="f1_macro", ascending=False)

    caminho_resultados = f"{OUTPUTS_DIR}/resultados_modelos.csv"
    resultados_df.to_csv(caminho_resultados, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("Comparação geral dos modelos")
    print("=" * 60)
    print(resultados_df)

    melhor_modelo_nome = resultados_df.iloc[0]["modelo"]
    melhor_modelo = modelos_treinados[melhor_modelo_nome]

    caminho_melhor_modelo = f"{MODELS_DIR}/melhor_modelo.pkl"
    joblib.dump(melhor_modelo, caminho_melhor_modelo)

    print("\nMelhor modelo escolhido:")
    print(melhor_modelo_nome)
    print(f"Melhor modelo salvo em: {caminho_melhor_modelo}")

    salvar_relatorio_texto(resultados_df, melhor_modelo_nome)

    print("\nEtapa 5 concluída com sucesso!")


if __name__ == "__main__":
    main()