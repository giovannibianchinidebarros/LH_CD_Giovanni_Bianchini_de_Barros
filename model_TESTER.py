# imports:
import pandas as pd
import joblib


# Função para converter os dados de entrada para o formato esperado para teste:
def transform_to_X_test(data):
    X_test = {
        'nome': [data.get('nome')],
        'room_type': [data.get('room_type')],
        'bairro': [data.get('bairro')],
        'bairro_group': [data.get('bairro_group')],
        'minimo_noites': [data.get('minimo_noites')],
        'numero_de_reviews': [data.get('numero_de_reviews')],
        'reviews_por_mes': [data.get('reviews_por_mes')],
        'calculado_host_listings_count': [data.get('calculado_host_listings_count')],
        'disponibilidade_365': [data.get('disponibilidade_365')]
    }
    return pd.DataFrame(X_test)


# Carregar o arquivo .pkl:
model = joblib.load('LH_CD_Giovanni_Bianchini_de_Barros.pkl')


# Teste:
teste = transform_to_X_test({'id': 2595,
                             'nome': 'Skylit Midtown Castle',
                             'host_id': 2845,
                             'host_name': 'Jennifer',
                             'bairro_group': 'Manhattan',
                             'bairro': 'Midtown',
                             'latitude': 40.75362,
                             'longitude': -73.98377,
                             'room_type': 'Entire home/apt',
                             'price': 225,
                             'minimo_noites': 1,
                             'numero_de_reviews': 45,
                             'ultima_review': '2019-05-21',
                             'reviews_por_mes': 0.38,
                             'calculado_host_listings_count': 2,
                             'disponibilidade_365': 355})


# Fazendo a previsão:
prediction = model.predict(teste)
print(prediction)
