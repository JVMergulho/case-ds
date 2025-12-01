import pandas as pd
import numpy as np

def calculate_psi(expected: pd.Series, actual: pd.Series, bins=10):
    """Calcula o Population Stability Index (PSI) para uma variável.

    Args:
        expected (pd.Series): Série de dados de referência (ex: Treino).
        actual (pd.Series): Série de dados atual (ex: OOT/Teste).
        bins (int or sequence): Número de bins (decis) ou lista com limites dos bins.
                                Para categóricas, este argumento é ignorado.

    Returns:
        float: O valor do PSI. Retorna 0 se a variável for constante.
    """
    
    # Se a variável tiver apenas um valor único, não há distribuição para comparar.
    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return 0.0

    # Lida com tipos diferentes
    if pd.api.types.is_numeric_dtype(expected):
        # --- Variáveis Numéricas ---
        
        # 1. Define os BINS com base APENAS nos dados esperados (treino)
        # Usamos qcut para criar decis (quantis)
        try:
            # Garante que os limites sejam únicos
            breaks, bins_edges = pd.qcut(expected, q=bins, retbins=True, duplicates='drop')
        except ValueError:
             # Se qcut falhar (poucos valores únicos), use cut normal
            breaks, bins_edges = pd.cut(expected, bins=bins, retbins=True, duplicates='drop')

        # 2. Aplica esses mesmos BINS aos dados atuais (OOT)
        expected_binned = pd.cut(expected, bins=bins_edges, include_lowest=True)
        actual_binned = pd.cut(actual, bins=bins_edges, include_lowest=True)

        # 3. Calcula as porcentagens por bin
        df_expected_perc = expected_binned.value_counts(normalize=True).reset_index()
        df_expected_perc.columns = ['bin', 'perc_expected']

        df_actual_perc = actual_binned.value_counts(normalize=True).reset_index()
        df_actual_perc.columns = ['bin', 'perc_actual']

        # 4. Junta as porcentagens, alinhando pelos bins
        psi_df = pd.merge(df_expected_perc, df_actual_perc, on='bin', how='outer')

    else:
        # --- Variáveis Categóricas ---
        
        # 1. Calcula as porcentagens por categoria
        df_expected_perc = expected.value_counts(normalize=True).reset_index()
        df_expected_perc.columns = ['bin', 'perc_expected']

        df_actual_perc = actual.value_counts(normalize=True).reset_index()
        df_actual_perc.columns = ['bin', 'perc_actual']

        # 2. Junta as porcentagens, alinhando pelas categorias
        psi_df = pd.merge(df_expected_perc, df_actual_perc, on='bin', how='outer')

    # Evitar divisão por zero ou log de zero - Adiciona um valor muito pequeno (epsilon)
    epsilon = 1e-6
    psi_df['perc_expected'] = psi_df['perc_expected'].replace(0, epsilon)
    psi_df['perc_actual'] = psi_df['perc_actual'].replace(0, epsilon)

    # Calcula o PSI
    psi_df['psi'] = (psi_df['perc_actual'] - psi_df['perc_expected']) * np.log(psi_df['perc_actual'] / psi_df['perc_expected'])

    # Soma os componentes para obter o PSI final
    psi_value = psi_df['psi'].sum()

    return psi_value

def analyze_psi(df_train, df_test, columns_to_check=None):
    """
    Calcula e interpreta o PSI para múltiplas colunas entre treino e teste.

    Args:
        df_train (pd.DataFrame): DataFrame de treino (referência).
        df_test (pd.DataFrame): DataFrame de teste/OOT (atual).
        columns_to_check (list, optional): Lista de colunas para analisar. 
                                           Se None, analisa todas as colunas comuns.

    Returns:
        pd.DataFrame: DataFrame com o PSI e interpretação para cada coluna.
    """
    if columns_to_check is None:
        columns_to_check = [col for col in df_train.columns if col in df_test.columns]

    results = []
    print(f"Calculando PSI para {len(columns_to_check)} colunas...")

    for col in columns_to_check:
        # Pula colunas de data/ID se não foram removidas antes
        if pd.api.types.is_datetime64_any_dtype(df_train[col]) or '_id' in col.lower():
            continue
            
        psi = calculate_psi(df_train[col], df_test[col])
        
        # Interpretação
        if psi < 0.1:
            interpretation = 'Estável'
        elif psi < 0.25:
            interpretation = 'Alerta'
        else:
            interpretation = 'Instável'
            
        results.append({'Coluna': col, 'PSI': psi, 'Status': interpretation})

    psi_results_df = pd.DataFrame(results)
    psi_results_df = psi_results_df.sort_values('PSI', ascending=False) # Ordena para ver os piores primeiro

    return psi_results_df