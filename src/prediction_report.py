import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, precision_score, recall_score
from IPython.display import display

blue = '#006e9cff'
red = "#cd6454ff"
gray = '#647f8f'
cyan = '#18dbdbff'

def plotProbHistograms(y_pred_proba, y_test, ax):
  """
  Gera e exibe histogramas das probabilidades previstas para múltiplos modelos
  em um eixo matplotlib específico.

  Args:
      y_pred_proba (np.array): Probabilidades previstas para a classe positiva (1).
      y_test (pd.Series): Valores verdadeiros da target.
      ax (matplotlib.axes.Axes): O eixo onde os histogramas serão desenhados.
  """

  df_plot = pd.DataFrame({
      'Probabilidade': y_pred_proba,
      'Classe': y_test.astype(int)
  })

  # Para cada classe, faz um subplot no eixo fornecido
  sns.histplot(
      data=df_plot[df_plot['Classe'] == 0],
      x='Probabilidade',
      bins=30,
      stat='density',
      color=blue,
      label='Classe 0 (Não Inadimplente)',
      kde=True, # Adicionar Kernel Density Estimate para suavizar o histograma
      ax=ax # Especifica o eixo para plotar
  )
  sns.histplot(
      data=df_plot[df_plot['Classe'] == 1],
      x='Probabilidade',
      bins=30,
      stat='density',
      color=red,
      label='Classe 1 (Inadimplente)',
      kde=True, # Adicionar Kernel Density Estimate para suavizar o histograma
      ax=ax # Especifica o eixo para plotar
  )

  ax.set_title('Distribuição das Probabilidades Previstas por Classe', fontsize=14)
  ax.set_xlabel('Probabilidade Prevista')
  ax.set_ylabel('Densidade')
  ax.legend()
  ax.grid(True)

def plot_feature_importance(model, feature_names, ax=None):
  if ax is None:
    fig, ax = plt.subplots(figsize=(8, 6))

  feature_importances = get_feature_importance_df(model, feature_names).head(10)

  # Plotar a importância das features
  sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax, color=cyan)
  ax.set_title('Importância das Features')
  ax.set_xlabel('Importância')
  ax.set_ylabel('Feature')

def get_feature_importance_df(model, feature_names):
  # Extrair a importância das features
  importances = model.feature_importances_
  feature_names = feature_names
  feature_importances = pd.DataFrame({
      'Feature': feature_names,
      'Importance': importances
  }).sort_values(by='Importance', ascending=False)
  
  return feature_importances

def evaluate_metrics(y_test, y_pred_proba):
    """
    Calcula as principais métricas de avaliação para um modelo de classificação binária.
    Retorna um DataFrame com os valores.
    """

    # AUC ROC
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Curva PR e PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc_score = auc(recall_curve, precision_curve)

    # Estatística KS (corrigida)
    df_ks = pd.DataFrame({'proba': y_pred_proba, 'target': y_test})
    df_ks = df_ks.sort_values('proba', ascending=False).reset_index(drop=True)

    df_ks['cum_pos'] = (df_ks['target'] == 1).cumsum() / (df_ks['target'] == 1).sum()
    df_ks['cum_neg'] = (df_ks['target'] == 0).cumsum() / (df_ks['target'] == 0).sum()
    df_ks['ks'] = df_ks['cum_pos'] - df_ks['cum_neg']

    ks_score = df_ks['ks'].max()
    ks_index = df_ks['ks'].idxmax()
    ks_threshold = df_ks['proba'].iloc[ks_index]

    # Precision e Recall no threshold KS
    precision = precision_score(y_test, y_pred_proba >= ks_threshold)
    recall = recall_score(y_test, y_pred_proba >= ks_threshold)

    metrics = pd.DataFrame({
        'Métrica': [
            'AUC (ROC)',
            'PR-AUC',
            'KS Statistic',
            'KS Threshold',
            'Precision (no KS)',
            'Recall (no KS)'
        ],
        'Valor': [
            round(auc_score, 4),
            round(pr_auc_score, 4),
            round(ks_score, 4),
            round(ks_threshold, 4),
            round(precision, 4),
            round(recall, 4)
        ]
    })

    return metrics

def plot_ks_curve(y_true, y_probas_positive, title='Curva KS', ax=None):
    """
    Plota a curva KS (Kolmogorov-Smirnov) para um modelo de classificação binária.
    A visualização do KS foi melhorada para ser visível mesmo com valores muito baixos.
    """
    # 1. Cria um DataFrame com os dados (sem alterações)
    df = pd.DataFrame({
        'y_true': y_true,
        'y_proba': y_probas_positive
    })
    df = df.sort_values(by='y_proba').reset_index(drop=True)

    # 2. Calcula as distribuições acumuladas (sem alterações)
    total_positives = df['y_true'].sum()
    total_negatives = len(df) - total_positives
    
    # Adicionado um pequeno valor (epsilon) para evitar divisão por zero se uma classe não estiver presente
    if total_positives == 0 or total_negatives == 0:
        print("Aviso: Dados de entrada contêm apenas uma classe. A curva KS não pode ser calculada.")
        return 0, 0, ax
        
    df['cdf_positive'] = df['y_true'].cumsum() / total_positives
    df['cdf_negative'] = (1 - df['y_true']).cumsum() / total_negatives

    # 3. Calcula o KS (sem alterações)
    df['ks'] = abs(df['cdf_positive'] - df['cdf_negative'])
    ks_statistic = df['ks'].max()
    
    best_threshold_row = df.loc[df['ks'].idxmax()]
    best_threshold = best_threshold_row['y_proba']
    y_positive_at_ks = best_threshold_row['cdf_positive']
    y_negative_at_ks = best_threshold_row['cdf_negative']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 5. Plot das curvas e do KS (ESTA PARTE FOI ALTERADA)
    ax.plot(df['y_proba'], df['cdf_positive'], label='CDF Positivos (Classe 1)', color=blue)
    ax.plot(df['y_proba'], df['cdf_negative'], label='CDF Negativos (Classe 0)', color=red)

    ax.plot([best_threshold, best_threshold], [y_negative_at_ks, y_positive_at_ks],
            linestyle='--', color='red', label=f'KS = {ks_statistic:.4f}')

    # Adiciona uma anotação com uma seta para mostrar a distância exata
    ax.annotate(
        f'KS = {ks_statistic:.4f}',
        xy=(best_threshold, y_negative_at_ks),
        xytext=(best_threshold + 0.1, y_negative_at_ks + 0.2), # Posição do texto
        arrowprops=dict(facecolor='black', shrink=0.05),
        horizontalalignment='center',
        verticalalignment='bottom'
    )
    
    # 6. Estilização do gráfico (sem alterações)
    ax.set_title(title)
    ax.set_xlabel('Probabilidade (Score)')
    ax.set_ylabel('Percentual Acumulado')
    ax.legend()
    ax.grid(True)
    
    return ks_statistic, best_threshold


def show_report(model, experiment_name, feature_names, y_test, y_pred_proba, show_charts=True):
    """
    Gera um dashboard 2x2 com as principais métricas de avaliação para um modelo de classificação binária.
    Inclui AUC, PR-AUC, KS e distribuição de probabilidades.
    """

    # 1. CÁLCULO DAS MÉTRICAS
    metrics = evaluate_metrics(y_test, y_pred_proba)
    display(metrics)

    if not show_charts:
        return

    auc_score = metrics.loc[metrics['Métrica'] == 'AUC (ROC)', 'Valor'].values[0]

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    # 2. DASHBOARD 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Relatório de Performance - {experiment_name}", fontsize=18)

    # --- Gráfico 1: Curva ROC ---
    ax1 = axes[0, 0]
    ax1.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}', color= blue)
    ax1.plot([0, 1], [0, 1], color=gray, linestyle='--')
    ax1.set_title('Curva ROC', fontsize=14)
    ax1.set_xlabel('FPR (Taxa de Falsos Positivos)')
    ax1.set_ylabel('TPR (Taxa de Verdadeiros Positivos)')
    ax1.legend()
    ax1.grid(True)

    # --- Gráfico 2: Curva KS ---
    from matplotlib.ticker import PercentFormatter
    ax2 = axes[0, 1]
    ks_value, best_threshold = plot_ks_curve(y_test, y_pred_proba, ax=ax2)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_title(f'Curva KS (KS={ks_value:.2%})', fontsize=14)

    # --- Gráfico 3: Distribuição das Probabilidades ---
    ax3 = axes[1, 0]
    plotProbHistograms(y_pred_proba, y_test, ax=ax3)
    ax3.set_title('Distribuição das Probabilidades', fontsize=14)

    # --- Gráfico 4: Importância das Features ---
    ax4 = axes[1, 1]
    plot_feature_importance(model, feature_names, ax=ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
