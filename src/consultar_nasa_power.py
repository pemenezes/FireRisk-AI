import requests
from datetime import datetime


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def validar_data(data):
    """
    Valida data no formato YYYYMMDD.
    Exemplo: 20240115
    """
    try:
        datetime.strptime(data, "%Y%m%d")
    except ValueError:
        raise ValueError(
            f"Data inválida: {data}. Use o formato YYYYMMDD, exemplo: 20240115."
        )


def consultar_nasa_power(latitude, longitude, data_inicio, data_fim):
    """
    Consulta dados meteorológicos reais da NASA POWER.

    Parâmetros utilizados:
    - T2M: temperatura média a 2 metros, em °C
    - RH2M: umidade relativa a 2 metros, em %
    - PRECTOTCORR: precipitação corrigida, em mm/dia
    - WS10M: velocidade do vento a 10 metros, em m/s
    """

    validar_data(data_inicio)
    validar_data(data_fim)

    parametros = "T2M,RH2M,PRECTOTCORR,WS10M"

    params = {
        "parameters": parametros,
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": data_inicio,
        "end": data_fim,
        "format": "JSON"
    }

    response = requests.get(NASA_POWER_URL, params=params, timeout=30)

    if response.status_code != 200:
        raise Exception(
            f"Erro ao consultar NASA POWER. "
            f"Status code: {response.status_code}. "
            f"Resposta: {response.text}"
        )

    dados = response.json()

    try:
        parametros_api = dados["properties"]["parameter"]

        temperatura = parametros_api["T2M"]
        umidade = parametros_api["RH2M"]
        precipitacao = parametros_api["PRECTOTCORR"]
        vento = parametros_api["WS10M"]

    except KeyError as erro:
        raise KeyError(
            f"Resposta inesperada da API. Campo não encontrado: {erro}. "
            f"Resposta completa: {dados}"
        )

    datas_disponiveis = list(temperatura.keys())

    if not datas_disponiveis:
        raise ValueError("A API não retornou dados para o período informado.")

    resultados = []

    for data in datas_disponiveis:
        velocidade_vento_ms = vento[data]
        velocidade_vento_kmh = velocidade_vento_ms * 3.6

        linha = {
            "data": data,
            "latitude": latitude,
            "longitude": longitude,
            "temperatura_media": round(float(temperatura[data]), 2),
            "umidade_relativa": round(float(umidade[data]), 2),
            "precipitacao_mm": round(float(precipitacao[data]), 2),
            "velocidade_vento": round(float(velocidade_vento_kmh), 2)
        }

        resultados.append(linha)

    return resultados


def main():
    latitude = -15.60
    longitude = -56.10

    data_inicio = "20240901"
    data_fim = "20240907"

    print("Consultando NASA POWER...")
    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")
    print(f"Período: {data_inicio} até {data_fim}")
    print()

    resultados = consultar_nasa_power(
        latitude=latitude,
        longitude=longitude,
        data_inicio=data_inicio,
        data_fim=data_fim
    )

    print("Dados retornados pela API:")
    for linha in resultados:
        print(linha)

    ultimo_registro = resultados[-1]

    print("\nÚltimo registro formatado para o FireRisk AI:")
    print(f"Temperatura média: {ultimo_registro['temperatura_media']} °C")
    print(f"Umidade relativa: {ultimo_registro['umidade_relativa']} %")
    print(f"Precipitação: {ultimo_registro['precipitacao_mm']} mm")
    print(f"Velocidade do vento: {ultimo_registro['velocidade_vento']} km/h")


if __name__ == "__main__":
    main()