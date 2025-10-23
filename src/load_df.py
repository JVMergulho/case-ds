import pandas as pd
from pathlib import Path

def load_df(local_path: Path | str) -> pd.DataFrame | None:
    """"
    Carrega um DataFrame a partir de um arquivo local (CSV ou Parquet).
    O tipo de arquivo é inferido pela extensão do arquivo.
    """

    local_path = Path(local_path) if isinstance(local_path, str) else local_path
    
    # Inferir o tipo de arquivo pela extensão
    ext = local_path.suffix.lower()
    if ext == '.parquet':
        file_type = 'parquet'
    elif ext == '.csv':
        file_type = 'csv'
    else:
        print(f"Extensão de arquivo não suportada: {ext}")
        return None

    # Ler arquivo
    try:
        if file_type == 'parquet':
            return pd.read_parquet(local_path, engine='fastparquet')
        elif file_type == 'csv':
            return pd.read_csv(local_path, delimiter=';')
    except Exception as e:
        print(f"Erro ao ler o arquivo {local_path}: {e}")
        return None