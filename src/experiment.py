import joblib
import os
from . import prediction_report
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.base import clone

def fit_and_predict(model, X_train, y_train, X_test):

  # Treinar o modelo
  model.fit(X_train, y_train)

  # Fazer previsões no conjunto de teste
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  y_pred = model.predict(X_test)

  return y_pred_proba, y_pred

def save_model_and_metrics(model, metrics, experiment_name, dir_path):

    model_package = {
        'model': model,
        'metrics': metrics
    }

    os.makedirs(dir_path, exist_ok=True)

    # Define o caminho completo do arquivo
    model_path = os.path.join(dir_path, f"{experiment_name}.joblib")

    joblib.dump(model_package, model_path)

    print(f"Modelo salvo em: {model_path}")

def run_experiment(model,
                   X_train, y_train, X_test, y_test, 
                   show_charts=True, save_model=True,
                    experiment_name = 'experiment',
                   dir_path = 'models'):
    ''''
    Executa o experimento completo: 
        - treina o modelo com X_train e y_train, 
        - avalia o modelo com X_test e y_test, 
        - exibe o relatório 
        - salva o modelo e métricas.
    '''

    try:
        # 1. Tenta verificar se o modelo já foi treinado
        check_is_fitted(model)
        
        # 2. Se NÃO deu erro, o modelo ESTÁ treinado.
        print(f"Modelo '{model.__class__.__name__}' detectado como 'treinado'.")
        print("Clonando modelo (com os mesmos hiperparâmetros) para novo treinamento...")
        model_to_train = clone(model) # Cria uma nova instância NÃO TREINADA
    except NotFittedError:
        # 3. Se deu erro, o modelo NÃO ESTÁ treinado (o que é esperado).
        print(f"Modelo '{model.__class__.__name__}' detectado como 'não treinado'.")
        print("Usando a instância original para o treinamento...")
        model_to_train = model #

    feature_names = X_train.columns.tolist()
    y_pred_proba, y_pred = fit_and_predict(model_to_train, X_train, y_train, X_test)
    metrics = prediction_report.evaluate_metrics(y_test, y_pred_proba)

    prediction_report.show_report(model_to_train, experiment_name, feature_names, y_test, y_pred_proba, show_charts=show_charts)

    if save_model:
        save_model_and_metrics(model_to_train, metrics, experiment_name, dir_path)