# 🔥 FireRisk AI

Sistema Inteligente de Previsão de Risco de Incêndios com Dados Orbitais

## 📌 Sobre o projeto

O **FireRisk AI** é uma solução de Inteligência Artificial desenvolvida para prever o risco de incêndios em determinada região a partir de variáveis ambientais, climáticas, históricas e orbitais simuladas.

O projeto foi desenvolvido no contexto da **Global Solution 2026 — Space Connect / Indústria Espacial**, com foco na aplicação de dados espaciais e técnicas de Machine Learning para resolver problemas reais na Terra.

A proposta do FireRisk AI é apoiar ações preventivas de órgãos públicos, Defesa Civil, brigadas ambientais, produtores rurais e gestores de áreas protegidas, permitindo a identificação antecipada de regiões com maior risco de incêndio.

## 🚀 Aplicação publicada

A aplicação está disponível no Streamlit Cloud:

🔗 https://firerisk-ai.streamlit.app

## 📂 Repositório

Código-fonte disponível em:

🔗 https://github.com/pemenezes/FireRisk-AI/tree/main

## 🎯 Problema

Incêndios florestais e queimadas causam impactos ambientais, sociais e econômicos significativos. Muitas regiões só conseguem reagir quando o fogo já está ativo, dificultando o controle e aumentando os prejuízos.

A ausência de ferramentas acessíveis de previsão e explicação do risco dificulta a tomada de decisão preventiva. Por isso, o FireRisk AI busca antecipar o risco de incêndios com base em dados ambientais e orbitais simulados, oferecendo uma classificação clara e recomendações preventivas.

## 🛰️ Conexão com a Indústria Espacial

O projeto se conecta à Indústria Espacial por meio do uso conceitual de dados orbitais e sensoriamento remoto, como:

* focos de calor detectados por satélite;
* índice de vegetação;
* localização geográfica;
* dados climáticos e ambientais;
* variáveis associadas ao monitoramento territorial.

Nesta versão MVP, os dados foram gerados de forma sintética e realista para validar o pipeline completo de IA. Em uma evolução futura, a solução poderá ser integrada a APIs reais, como NASA FIRMS, INPE Queimadas, NASA POWER e dados de satélites Sentinel/Copernicus.

## 🌱 ODS relacionados

O FireRisk AI está alinhado principalmente aos seguintes Objetivos de Desenvolvimento Sustentável:

* **ODS 9 — Indústria, inovação e infraestrutura**
* **ODS 11 — Cidades e comunidades sustentáveis**
* **ODS 13 — Ação contra a mudança global do clima**
* **ODS 2 — Fome zero e agricultura sustentável**, considerando a proteção de áreas agrícolas contra incêndios

## 🧠 Objetivo da IA

O objetivo do modelo é classificar o risco de incêndio de uma região em quatro níveis:

* `baixo`
* `medio`
* `alto`
* `critico`

O problema foi tratado como uma tarefa de **classificação multiclasse**.

## 🗃️ Dataset

Foi criado um dataset sintético realista contendo:

* **1.500 registros**
* **15 colunas**
* **14 variáveis de entrada**
* **1 variável alvo**

A variável alvo é:

```text
risco_incendio
```

### Colunas utilizadas

| Coluna                | Descrição                                       |
| --------------------- | ----------------------------------------------- |
| `latitude`            | Latitude da região analisada                    |
| `longitude`           | Longitude da região analisada                   |
| `estado`              | Estado brasileiro                               |
| `bioma`               | Bioma predominante da região                    |
| `mes`                 | Mês da análise                                  |
| `temperatura_media`   | Temperatura média em °C                         |
| `umidade_relativa`    | Umidade relativa do ar em %                     |
| `precipitacao_mm`     | Precipitação acumulada em milímetros            |
| `velocidade_vento`    | Velocidade do vento em km/h                     |
| `dias_sem_chuva`      | Quantidade de dias consecutivos sem chuva       |
| `focos_calor_7d`      | Quantidade de focos de calor nos últimos 7 dias |
| `distancia_foco_km`   | Distância até o foco de calor mais próximo      |
| `ndvi`                | Índice de vegetação                             |
| `historico_incendios` | Histórico de incêndios na região                |
| `risco_incendio`      | Classe de risco prevista                        |

## ⚙️ Pipeline de Machine Learning

O pipeline desenvolvido contempla as seguintes etapas:

1. Geração do dataset sintético
2. Validação e análise inicial dos dados
3. Pré-processamento
4. Separação entre treino e teste
5. Treinamento de modelos preditivos
6. Comparação de métricas
7. Escolha do melhor modelo
8. Interpretabilidade com SHAP
9. Deploy da aplicação com Streamlit

## 🧪 Modelos utilizados

Foram treinados e comparados dois modelos de Machine Learning:

* **Random Forest Classifier**
* **Gradient Boosting Classifier**

A escolha do melhor modelo foi feita com base no **F1 Macro**, pois o dataset possui classes moderadamente desbalanceadas.

## 📊 Resultados obtidos

### Comparação geral dos modelos

| Modelo            | Acurácia | Precision Macro | Recall Macro | F1 Macro | F1 Weighted |
| ----------------- | -------: | --------------: | -----------: | -------: | ----------: |
| Random Forest     |   0.7600 |          0.8138 |       0.6703 |   0.7108 |      0.7497 |
| Gradient Boosting |   0.7600 |          0.7585 |       0.7447 |   0.7492 |      0.7588 |

O modelo escolhido foi o **Gradient Boosting Classifier**, pois apresentou melhor desempenho em **F1 Macro**.

### Resultado no conjunto de teste

| Métrica                     |  Valor |
| --------------------------- | -----: |
| Total de registros testados |    300 |
| Total de acertos            |    228 |
| Total de erros              |     72 |
| Acurácia                    | 0.7600 |

## 🔎 Interpretabilidade com SHAP

Foi utilizada a biblioteca **SHAP** para explicar quais variáveis mais influenciaram as decisões do modelo.

As principais variáveis identificadas foram:

| Posição | Variável              | Importância média SHAP |
| ------: | --------------------- | ---------------------: |
|       1 | `umidade_relativa`    |                 0.0640 |
|       2 | `precipitacao_mm`     |                 0.0569 |
|       3 | `dias_sem_chuva`      |                 0.0523 |
|       4 | `temperatura_media`   |                 0.0477 |
|       5 | `focos_calor_7d`      |                 0.0466 |
|       6 | `distancia_foco_km`   |                 0.0419 |
|       7 | `mes`                 |                 0.0328 |
|       8 | `ndvi`                |                 0.0314 |
|       9 | `historico_incendios` |                 0.0262 |
|      10 | `velocidade_vento`    |                 0.0136 |

A análise SHAP mostrou que o modelo priorizou variáveis tecnicamente coerentes com o risco de incêndio, como baixa umidade, pouca precipitação, muitos dias sem chuva, temperatura elevada e focos de calor próximos.

## 🖥️ Aplicação Streamlit

A aplicação permite:

* carregar um cenário de baixo risco;
* carregar um cenário crítico;
* informar manualmente variáveis ambientais e regionais;
* prever o risco de incêndio;
* visualizar probabilidades por classe;
* receber recomendações preventivas;
* visualizar a importância das variáveis com SHAP.

## 🏗️ Estrutura do projeto

```text
FireRisk-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── firerisk_dataset.csv
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── gradient_boosting_pipeline.pkl
│   ├── melhor_modelo.pkl
│   └── random_forest_pipeline.pkl
│
├── outputs/
│   ├── figures/
│   ├── shap/
│   │   ├── explicacao_local_exemplo.csv
│   │   └── importancia_global_shap.csv
│   ├── relatorio_shap.txt
│   ├── relatorio_treinamento.txt
│   ├── relatorio_validacao.txt
│   └── resultados_modelos.csv
│
├── src/
│   ├── analisar_resultados.py
│   ├── explicar_shap.py
│   ├── gerar_dataset.py
│   ├── preprocessamento.py
│   └── treinar_modelos.py
│
├── .gitignore
├── packages.txt
├── requirements.txt
└── README.md
```

## ▶️ Como executar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/pemenezes/FireRisk-AI.git
cd FireRisk-AI
```

### 2. Criar ambiente virtual

```bash
python -m venv .venv
```

### 3. Ativar ambiente virtual

No Windows:

```bash
.venv\Scripts\activate
```

No Git Bash:

```bash
source .venv/Scripts/activate
```

No Linux/Mac:

```bash
source .venv/bin/activate
```

### 4. Instalar dependências

```bash
python -m pip install -r requirements.txt
```

### 5. Executar a aplicação

```bash
python -m streamlit run app/app.py
```

## 🔁 Como reproduzir o pipeline completo

### 1. Gerar dataset

```bash
python src/gerar_dataset.py
```

### 2. Pré-processar dados

```bash
python src/preprocessamento.py
```

### 3. Treinar modelos

```bash
python src/treinar_modelos.py
```

### 4. Analisar resultados

```bash
python src/analisar_resultados.py
```

### 5. Gerar explicações SHAP

```bash
python src/explicar_shap.py
```

### 6. Executar aplicação

```bash
python -m streamlit run app/app.py
```

## 🧾 Principais arquivos gerados

| Arquivo                                    | Descrição                                       |
| ------------------------------------------ | ----------------------------------------------- |
| `data/firerisk_dataset.csv`                | Dataset sintético completo                      |
| `data/train.csv`                           | Dados de treino                                 |
| `data/test.csv`                            | Dados de teste                                  |
| `models/melhor_modelo.pkl`                 | Melhor modelo treinado                          |
| `outputs/resultados_modelos.csv`           | Comparação entre modelos                        |
| `outputs/relatorio_treinamento.txt`        | Relatório de treinamento                        |
| `outputs/relatorio_validacao.txt`          | Relatório de validação                          |
| `outputs/relatorio_shap.txt`               | Relatório de interpretabilidade                 |
| `outputs/shap/importancia_global_shap.csv` | Importância global das variáveis                |
| `outputs/figures/`                         | Gráficos de métricas, matriz de confusão e SHAP |

## 🧩 Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* SHAP
* Matplotlib
* Streamlit
* GitHub
* Streamlit Cloud

## 📌 Limitações do MVP

Esta versão utiliza um dataset sintético realista. Isso permite validar o pipeline completo de IA de ponta a ponta, mas ainda não representa uma operação em tempo real com dados oficiais.

As principais limitações são:

* os dados não são coletados em tempo real;
* o dataset foi gerado artificialmente;
* não há integração automática com APIs externas;
* o modelo ainda não foi validado com bases reais históricas;
* o sistema não substitui análises técnicas de órgãos oficiais.

## 🔮 Próximos passos

Como evolução futura, o FireRisk AI poderá ser integrado a fontes reais, como:

* NASA FIRMS;
* INPE Queimadas;
* NASA POWER;
* Sentinel/Copernicus;
* bases meteorológicas públicas;
* dados históricos de focos de calor e queimadas.

Também poderão ser adicionadas funcionalidades como:

* mapa interativo;
* alertas automáticos;
* consulta por município;
* atualização diária dos dados;
* integração com dashboards;
* API para consumo por outros sistemas;
* priorização automática de áreas críticas.

## 👥 Integrantes

```text
pedro henrique menezes - RM97432
Pedro Gava - RM551043
Henrique Lima - RM551528
Anny Dias - RM98295
```

## 📚 Contexto acadêmico

Projeto desenvolvido para a **Global Solution 2026 — Engenharia de Software — FIAP**, com foco na disciplina **Generative AI For Engineering**.

## ✅ Status

Projeto funcional com:

* dataset sintético realista;
* pipeline completo de Machine Learning;
* dois modelos comparados;
* validação com métricas;
* interpretabilidade com SHAP;
* aplicação publicada no Streamlit Cloud;
* repositório público no GitHub.
