import pandas as pd
from pathlib import Path

# Função para converter colunas de um DataFrame
def convert_to_datetime(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """
    Converte uma lista de colunas em um DataFrame para o tipo datetime.
    """
    print(f"\n--- Verificando tipos de dados ANTES da conversão ---")
    print(df[colunas].dtypes)

    for coluna in colunas:
        if coluna in df.columns:
            df[coluna] = pd.to_datetime(df[coluna], errors='coerce')
        else:
            print(f"Aviso: A coluna '{coluna}' não foi encontrada no DataFrame.")

    print(f"\n--- Verificando tipos de dados DEPOIS da conversão ---")
    print(df[colunas].dtypes)

    return df