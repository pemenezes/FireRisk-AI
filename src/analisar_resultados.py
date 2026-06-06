import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


TARGET = "risco_incendio"

TEST_PATH = "data/test.csv"
RESULTADOS_MODELOS_PATH = "outputs/resultados_modelos.csv"
MELHOR_MODELO_PATH = "models/melhor_modelo.pkl"

OUTPUTS_DIR = "outputs"
FIGURES_DIR = "outputs/figures"

CLASSES = ["baixo", "medio", "alto", "critico"]


def carregar_arquivos():
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {TEST_PATH}")

    if not os.path.exists(RESULTADOS_MODELOS_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {RESULTADOS_MODELOS_PATH}")

    if not os.path.exists(MELHOR_MODELO_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {MELHOR_MODELO_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    resultados_df = pd.read_csv(RESULTADOS_MODELOS_PATH)
    modelo = joblib.load(MELHOR_MODELO_PATH)

    return test_df, resultados_df, modelo


def gerar_grafico_comparacao_metricas(resultados_df):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metricas = ["acuracia", "f1_macro", "f1_weighted"]

    df_plot = resultados_df[["modelo"] + metricas].copy()
    df_plot = df_plot.set_index("modelo")

    ax = df_plot.plot(kind="bar", figsize=(10, 6))

    plt.title("Comparação de Métricas dos Modelos")
    plt.xlabel("Modelo")
    plt.ylabel("Valor da métrica")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Métrica")
    plt.tight_layout()

    caminho_saida = f"{FIGURES_DIR}/comparacao_metricas_modelos.png"
    plt.savefig(caminho_saida)
    plt.close()

    print(f"Gráfico de comparação salvo em: {caminho_saida}")


def gerar_previsoes(modelo, test_df):
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    y_pred = modelo.predict(X_test)

    previsoes_df = test_df.copy()
    previsoes_df["risco_previsto"] = y_pred
    previsoes_df["acertou"] = previsoes_df[TARGET] == previsoes_df["risco_previsto"]

    caminho_previsoes = f"{OUTPUTS_DIR}/previsoes_teste.csv"
    previsoes_df.to_csv(caminho_previsoes, index=False, encoding="utf-8")

    erros_df = previsoes_df[previsoes_df["acertou"] == False].copy()
    caminho_erros = f"{OUTPUTS_DIR}/erros_modelo.csv"
    erros_df.to_csv(caminho_erros, index=False, encoding="utf-8")

    print(f"Previsões salvas em: {caminho_previsoes}")
    print(f"Erros do modelo salvos em: {caminho_erros}")

    return y_test, y_pred, previsoes_df, erros_df


def analisar_erros(erros_df):
    if erros_df.empty:
        return "O modelo não apresentou erros no conjunto de teste."

    analise = []

    analise.append("Quantidade de erros por classe real:")
    analise.append(str(erros_df[TARGET].value_counts()))

    analise.append("\nQuantidade de erros por classe prevista:")
    analise.append(str(erros_df["risco_previsto"].value_counts()))

    analise.append("\nCruzamento entre classe real e classe prevista:")
    cruzamento = pd.crosstab(
        erros_df[TARGET],
        erros_df["risco_previsto"],
        rownames=["Classe real"],
        colnames=["Classe prevista"]
    )
    analise.append(str(cruzamento))

    return "\n".join(analise)


def salvar_relatorio_validacao(resultados_df, y_test, y_pred, previsoes_df, erros_df):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    caminho_saida = f"{OUTPUTS_DIR}/relatorio_validacao.txt"

    acuracia = accuracy_score(y_test, y_pred)
    total_testes = len(previsoes_df)
    total_acertos = int(previsoes_df["acertou"].sum())
    total_erros = len(erros_df)
    taxa_erro = total_erros / total_testes

    matriz = confusion_matrix(y_test, y_pred, labels=CLASSES)

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        arquivo.write("Relatório de Validação - FireRisk AI\n")
        arquivo.write("=" * 50)
        arquivo.write("\n\n")

        arquivo.write("1. Comparação geral dos modelos\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write(resultados_df.to_string(index=False))
        arquivo.write("\n\n")

        arquivo.write("2. Melhor modelo utilizado\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write("O melhor modelo utilizado nesta validação foi o modelo salvo em models/melhor_modelo.pkl.\n")
        arquivo.write("A escolha foi feita com base no maior F1 Macro, pois o dataset possui classes moderadamente desbalanceadas.\n\n")

        arquivo.write("3. Resultados no conjunto de teste\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write(f"Total de registros de teste: {total_testes}\n")
        arquivo.write(f"Total de acertos: {total_acertos}\n")
        arquivo.write(f"Total de erros: {total_erros}\n")
        arquivo.write(f"Acurácia: {acuracia:.4f}\n")
        arquivo.write(f"Taxa de erro: {taxa_erro:.4f}\n\n")

        arquivo.write("4. Relatório de classificação\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write(classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            zero_division=0
        ))
        arquivo.write("\n\n")

        arquivo.write("5. Matriz de confusão\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        matriz_df = pd.DataFrame(
            matriz,
            index=[f"Real: {classe}" for classe in CLASSES],
            columns=[f"Previsto: {classe}" for classe in CLASSES]
        )
        arquivo.write(matriz_df.to_string())
        arquivo.write("\n\n")

        arquivo.write("6. Análise dos erros\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write(analisar_erros(erros_df))
        arquivo.write("\n\n")

        arquivo.write("7. Interpretação técnica\n")
        arquivo.write("-" * 50)
        arquivo.write("\n")
        arquivo.write(
            "O modelo apresentou desempenho adequado para um MVP baseado em dados sintéticos realistas. "
            "Como o objetivo inicial é validar o pipeline completo de IA, os resultados são suficientes para prosseguir "
            "para a etapa de interpretabilidade com SHAP e posteriormente para o deploy da aplicação. "
            "Em uma evolução futura, o dataset poderá ser substituído ou complementado por dados reais de APIs "
            "como NASA FIRMS, INPE Queimadas e NASA POWER."
        )

    print(f"Relatório de validação salvo em: {caminho_saida}")


def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    test_df, resultados_df, modelo = carregar_arquivos()

    print("Arquivos carregados com sucesso.")
    print(f"Formato do conjunto de teste: {test_df.shape}")
    print(f"Modelos comparados: {list(resultados_df['modelo'])}")

    gerar_grafico_comparacao_metricas(resultados_df)

    y_test, y_pred, previsoes_df, erros_df = gerar_previsoes(modelo, test_df)

    salvar_relatorio_validacao(
        resultados_df,
        y_test,
        y_pred,
        previsoes_df,
        erros_df
    )

    print("\nResumo da validação:")
    print(f"Total de registros testados: {len(previsoes_df)}")
    print(f"Total de acertos: {int(previsoes_df['acertou'].sum())}")
    print(f"Total de erros: {len(erros_df)}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")

    print("\nEtapa 6 concluída com sucesso!")


if __name__ == "__main__":
    main()