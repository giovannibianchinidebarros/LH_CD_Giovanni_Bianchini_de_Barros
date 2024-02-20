
# LH_CD_Giovanni_Bianchini_de_Barros
### Desafio Cientista de Dados - Indicium 2024


Este é meu projeto para o Processo Seletivo do Programa **Lighthouse** da Indicium.

---

#### Sobre o dataset:
> Temos um conjunto de dados com 48.894 linhas e 16 colunas, que será utilizado para desenvolver uma estratégia de precificação para uma plataforma de aluguel temporário na cidade de Nova York. Como estamos fazendo uma previsão do preços, este é um problema de **regressão**.

#### Análise:
> Alguns dados cruciais para determinar o valor de um imóvel, como tamanho, número de quartos/banheiros, acabamento ou presença de mobília, não estão disponíveis neste conjunto de dados. Porém a coluna **'nome'** possui, em alguns casos, uma descrição do lugar.
Por isso decidi utilizar um modelo de reconhecimento de texto (NLP) para processar essa coluna.

> O conjunto de dados inclui a localização de cada imóvel. Optei por classificar os imóveis por **bairro** e descartar as colunas de **latitude** e **longitude**. Além disso, a coluna **'room_type'** possui três categorias bem definidas (Entire home/apt, Private room, Shared room), que certamente influenciam no preço. Estou utilizando o OneHotEncoder do sklearn.preprocessing para lidar com as colunas **'bairro_group'**, **'bairro'** e **'room_type'**.

> As demais colunas contêm valores numéricos. Durante a análise exploratória dos dados, não identifiquei uma relação muito significativa entre o **número mínimo de noites**, a **disponibilidade ao longo do ano** e o **número de reviews** com o **preço final**. Mesmo assim, esses dados serão incluídos no treinamento dos modelos.

> Percebi uma relação entre o **número de reviews** e o **número mínimo de noites**. Imóveis com um número mínimo de noites menor tendem a ter mais reviews, o que faz sentido, se pensar que esses imóveis são alugados com mais frequência. Seria interessante explorar estratégias de marketing focando nisso.

> A análise e códigos estão no arquivo: [LH_CD_Giovanni_Bianchini_de_Barros.ipynb](https://github.com/giovannibianchinidebarros/LH_CD_Giovanni_Bianchini_de_Barros/blob/main/LH_CD_Giovanni_Bianchini_de_Barros.ipynb)

#### Modelo escolhido:
> Testei diferentes modelos de regressão, incluindo Linear, Polinomial, Support Vector, Decision Tree e Random Forest Regression, avaliando todos com base no R2 Score. Nos testes, o modelo de **Random Forest Regression** demonstrou ser o mais eficaz.

>  O modelo criado está disponível neste repositório:  [LH_CD_Giovanni_Bianchini_de_Barros.pkl](https://github.com/giovannibianchinidebarros/LH_CD_Giovanni_Bianchini_de_Barros/blob/main/LH_CD_Giovanni_Bianchini_de_Barros.pkl)


#### Requisitos:
> **Arquivo de requisitos** com todos os pacotes utilizados e suas versões: [requirements.txt](https://github.com/giovannibianchinidebarros/LH_CD_Giovanni_Bianchini_de_Barros/blob/main/requirements.txt
)

---

#### Testes:

Para testar o Modelo, recomendo utilizar o seguinte código:

``` py
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
```

O mesmo código para teste se encontra em:

[model_TESTER.ipynb](https://github.com/giovannibianchinidebarros/LH_CD_Giovanni_Bianchini_de_Barros/blob/main/model_TESTER.ipynb)

[model_TESTER.py](https://github.com/giovannibianchinidebarros/LH_CD_Giovanni_Bianchini_de_Barros/blob/main/model_TESTER.py)

---