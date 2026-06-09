# 🛰️ ATLAS — FireRisk AI

Módulo de Inteligência Artificial para previsão de risco de queimadas com dados orbitais

## 📌 Sobre o projeto

O **ATLAS** é uma plataforma inteligente de monitoramento e resposta a desastres ambientais baseada em dados orbitais, sensoriamento remoto e Inteligência Artificial.

A solução foi pensada para apoiar principalmente a prevenção de **queimadas** e **deslizamentos**, dois problemas críticos que atingem com maior intensidade regiões vulneráveis do Brasil e exigem resposta rápida, precisa e acessível.

Dentro da arquitetura geral do ATLAS, este repositório implementa o **FireRisk AI**, um módulo de Inteligência Artificial focado na previsão de risco de incêndios e queimadas. O objetivo deste MVP é demonstrar como dados ambientais, climáticos, históricos e orbitais podem ser processados por modelos de Machine Learning para classificar o risco de incêndio em uma região.

A proposta do ATLAS vai além de apenas mapear ocorrências. Muitas soluções mostram o que já aconteceu e onde o evento ocorreu. O ATLAS busca transformar esse cenário em inteligência preditiva, permitindo identificar regiões com maior probabilidade de risco antes que o desastre aconteça.

## 🚀 Aplicação publicada

A aplicação está disponível no Streamlit Cloud:

https://firerisk-ai.streamlit.app

## 📂 Repositório

Código-fonte disponível em:

https://github.com/pemenezes/FireRisk-AI/tree/main

## 🎯 Problema

Incêndios florestais, queimadas e deslizamentos causam impactos humanos, ambientais e materiais significativos. Em muitos municípios, principalmente em regiões vulneráveis, a resposta a esses eventos ainda ocorre de forma reativa, quando o desastre já aconteceu.

Além disso, muitos órgãos locais não possuem equipes especializadas em análise geoespacial ou infraestrutura técnica para operar sistemas complexos de monitoramento.

O desafio do ATLAS é transformar dados orbitais e ambientais em informação útil para tomada de decisão, permitindo que equipes como Defesa Civil, prefeituras, brigadas ambientais e gestores públicos atuem de forma preventiva.

Neste MVP, o foco foi a previsão de **risco de queimadas**, classificando uma região em quatro níveis de criticidade:

* `baixo`
* `medio`
* `alto`
* `critico`

## 🛰️ Solução proposta: ATLAS

O **ATLAS** é uma plataforma de monitoramento e resposta a desastres ambientais que utiliza dados de satélites, sensoriamento remoto, dados climáticos e Inteligência Artificial para gerar previsões de risco por região.

A solução foi concebida para integrar fontes como:

* Sentinel-2;
* NASA FIRMS;
* INPE Queimadas;
* CEMADEN;
* NASA POWER;
* sensores locais e IoT em versões futuras.

A partir desses dados, o ATLAS pode apoiar a identificação de áreas com maior risco de queimadas e deslizamentos, gerando alertas classificados por nível de criticidade e recomendações operacionais em linguagem acessível.

Exemplo de alerta ideal da plataforma ATLAS:

```text
Setor Norte com 87% de risco de deslizamento nas próximas 36 horas.
Recomendação: acionar evacuação preventiva em comunidades do perímetro X.
```

No escopo deste repositório, foi implementado o módulo **FireRisk AI**, responsável pela previsão de risco de queimadas/incêndios.

## 🔥 Módulo implementado: FireRisk AI

O **FireRisk AI** é o módulo de IA do ATLAS voltado à previsão de risco de incêndios.

Ele utiliza variáveis ambientais, climáticas, históricas e orbitais para estimar o risco de incêndio de uma região. O modelo foi treinado para classificar o risco em quatro níveis: baixo, médio, alto e crítico.

A aplicação permite:

* carregar cenários simulados de baixo risco e risco crítico;
* inserir dados manualmente;
* consultar dados climáticos reais via NASA POWER;
* prever o risco de incêndio;
* visualizar probabilidades por classe;
* visualizar recomendações preventivas;
* consultar a importância das variáveis com SHAP;
* explorar o dataset utilizado no treinamento.

## 🧠 Objetivo da IA

O objetivo da Inteligência Artificial neste MVP é realizar uma tarefa de **classificação multiclasse**, prevendo o nível de risco de incêndio de uma região.

A variável alvo do modelo é:

```text
risco_incendio
```

As classes previstas são:

| Classe    | Significado                                 |
| --------- | ------------------------------------------- |
| `baixo`   | Condições pouco favoráveis para incêndio    |
| `medio`   | Atenção moderada                            |
| `alto`    | Risco relevante, exige monitoramento        |
| `critico` | Risco grave, exige alerta e ação preventiva |

## 🗃️ Dataset

Foi criado um dataset sintético realista para validar o pipeline completo de IA.

O dataset possui:

* **1.500 registros**
* **15 colunas**
* **14 variáveis de entrada**
* **1 variável alvo**

A geração sintética foi usada para permitir o funcionamento completo do MVP, mesmo antes da integração total com bases históricas reais.

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

## 🌎 Integração com dados reais — NASA POWER

Além do dataset sintético utilizado para treinamento do modelo, o FireRisk AI possui uma integração inicial com a API **NASA POWER**.

Essa integração permite consultar dados climáticos reais a partir de:

* latitude;
* longitude;
* data da consulta.

Os dados retornados pela API são usados para preencher automaticamente algumas variáveis da aplicação:

* temperatura média;
* umidade relativa;
* precipitação;
* velocidade do vento.

Essa funcionalidade aproxima o MVP da proposta real do ATLAS, demonstrando como dados externos e orbitais podem alimentar o processo de tomada de decisão.

## ⚙️ Pipeline de Machine Learning

O pipeline desenvolvido contempla as seguintes etapas:

1. Geração do dataset sintético realista
2. Validação e análise inicial dos dados
3. Pré-processamento
4. Separação entre treino e teste
5. Treinamento de modelos preditivos
6. Comparação de métricas
7. Escolha do melhor modelo
8. Interpretabilidade com SHAP
9. Deploy da aplicação com Streamlit
10. Integração inicial com API real da NASA POWER

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

A aplicação web funciona como uma versão inicial do **Painel Operacional ATLAS** para o módulo de queimadas.

Ela permite:

* visualizar o objetivo do projeto;
* carregar cenários prontos de baixo risco e risco crítico;
* consultar dados climáticos reais via NASA POWER;
* preencher dados ambientais e regionais;
* prever o risco de incêndio;
* visualizar probabilidades por classe;
* receber recomendações preventivas;
* visualizar a importância das variáveis via SHAP;
* explorar o dataset utilizado no treinamento.

## 🧭 Papel do FireRisk AI dentro do ATLAS

O FireRisk AI representa a primeira camada funcional da visão geral do ATLAS.

Na solução completa, o ATLAS poderia conter múltiplos módulos especializados:

| Módulo                   | Objetivo                                                 |
| ------------------------ | -------------------------------------------------------- |
| FireRisk AI              | Previsão de risco de queimadas/incêndios                 |
| LandslideRisk AI         | Previsão de risco de deslizamentos                       |
| Orbital Data Hub         | Integração de dados orbitais e ambientais                |
| Alert Engine             | Geração de alertas em linguagem natural                  |
| Painel Operacional ATLAS | Visualização de mapas, riscos e recomendações            |
| IoT/Sensor Layer         | Expansão com sensores locais, LoRaWAN e IoT via satélite |

Neste repositório, foi desenvolvido o módulo **FireRisk AI** como MVP funcional da camada de IA.

## 🌱 ODS relacionados

A solução se conecta diretamente aos seguintes Objetivos de Desenvolvimento Sustentável:

* **ODS 9 — Indústria, inovação e infraestrutura**
* **ODS 11 — Cidades e comunidades sustentáveis**
* **ODS 13 — Ação contra a mudança global do clima**

O ATLAS promove inovação tecnológica, apoia cidades mais resilientes e fortalece a resposta preventiva a eventos climáticos e ambientais extremos.

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
│   ├── consultar_nasa_power.py
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

### 6. Testar consulta NASA POWER

```bash
python src/consultar_nasa_power.py
```

### 7. Executar aplicação

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
* Requests
* NASA POWER API
* GitHub
* Streamlit Cloud

## 📌 Limitações do MVP

Esta versão implementa o módulo de previsão de risco de queimadas do ATLAS, mas ainda não representa a plataforma completa.

As principais limitações são:

* o treinamento ainda utiliza dataset sintético realista;
* a integração com dados reais está limitada à NASA POWER;
* focos de calor reais ainda não são consumidos automaticamente via NASA FIRMS ou INPE;
* o módulo de deslizamentos ainda não foi implementado;
* o painel ainda não possui mapas operacionais em tempo real;
* o sistema ainda não emite alertas automáticos;
* o modelo ainda não foi validado com uma base histórica oficial completa.

## 🔮 Próximos passos

Como evolução futura, o ATLAS poderá incorporar:

* integração com NASA FIRMS para focos de calor reais;
* integração com INPE Queimadas;
* integração com CEMADEN para risco hidrológico e deslizamentos;
* uso de imagens Sentinel-2;
* mapa interativo com regiões monitoradas;
* alertas automáticos por nível de criticidade;
* geração de alertas em linguagem natural;
* previsão de risco de deslizamentos;
* IoT via satélite e LoRaWAN para áreas remotas;
* API para consumo por órgãos públicos;
* painel operacional completo para Defesa Civil.

## 👥 Integrantes


```text
Anny Dias	RM98295
Henrique Lima	RM551528
Pedro Gava	RM551043
Pedro Menezes	RM97432
```

## 📚 Contexto acadêmico

Projeto desenvolvido para a **Global Solution 2026 — Engenharia de Software — FIAP**, dentro da proposta de soluções tecnológicas conectadas à Indústria Espacial.

Este repositório representa a entrega da camada de Inteligência Artificial, implementando o módulo **FireRisk AI** como parte da visão geral do **ATLAS**.

## ✅ Status

Projeto funcional com:

* dataset sintético realista;
* pipeline completo de Machine Learning;
* dois modelos comparados;
* validação com métricas;
* interpretabilidade com SHAP;
* aplicação publicada no Streamlit Cloud;
* visualização do dataset;
* integração inicial com dados climáticos reais via NASA POWER;
* repositório público no GitHub.
