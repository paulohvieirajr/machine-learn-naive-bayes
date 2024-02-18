import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.simplefilter('ignore')

# Coletando dados de risco da base.
df_risco = pd.read_csv('./dataset/base_risco.csv',
                       sep=';',
                       encoding='latin1')

print(df_risco)

# Selecionando as colunas que vamos utilizar para criar o indicador.
colunas_selecionadas = ['renda', 'tipo_divida', 'fiador', 'historico_credito']

# Criando um identificador para cada uma das variavies categoricas.
LE = LabelEncoder()

# Criando uma nova coluna baseada no encoder criado.
for coluna in colunas_selecionadas:
    df_risco[f'id_{coluna}'] = LE.fit_transform(df_risco[coluna])

# Organizando as colunas no dataframe
df_risco= df_risco[['id_renda', 'renda', 
                    'id_tipo_divida', 'tipo_divida', 
                    'id_fiador', 'fiador',
                    'id_historico_credito', 'historico_credito',
                    'risco']]

print(df_risco.head())

# Selecionando apenas as colunas que vamos utilizar no modelo.
colunas_modelo = [
    'id_renda', 'id_tipo_divida', 'id_fiador', 'id_historico_credito'
]

df_risco[colunas_modelo]

# Separando features do target.
X_dados = df_risco[colunas_modelo]
y_dados = df_risco['risco']

# Criação do modelo que fará o aprendizado.
model = GaussianNB()

# Fornecendo os dados a serem analisados (X) e o que se espera de resultado (Y). 
model.fit(X_dados.values, y_dados.values)

# Visualizando dados do modelo
# Quantidade de classes
print(f'Quantidade de classes encontradas: {model.classes_}')

# Contando as quantidades de itens por cada classe
print(f'Quantidade de itens por cada classe: {model.class_count_}')

# Porcentagem das quantidades por classe
print(f'Quantidade em porcentagem de itens por cada classe: {model.class_prior_}')

# Observando a porcentagem de acertos do modelo com base no aprendizado.
print(f'Porcentagem de acertos com base no aprendizado: {model.score(X_dados.values, y_dados.values)}')

# Criando coluna com as previsões que o modelo fez para os mesmos dados usados no treinamento.
df_risco['classe_predita'] = model.predict(X=X_dados.values)
print('Colocando na grid coluna com as previsões feitas pelo algoritmo:')
print(df_risco)

# Analisando um novo registro

# Renda acima de R$ 40.000
# Tipo de dívida Baixa
# Possui fiador
# Histórico de crédito Ruim

dados_cliente = [[0, 1, 1, 2]]
print(f'Previsão de risco com base em novos dados: {model.predict(dados_cliente)}')

# Criando novos registros para o algoritmo classificar.
novos_registros = [
    [1, 0, 1, 0], # Entre 13 e 40 , Divida alta, Possui fiador, Crédito bom
    [2, 1, 0, 0],
    [0, 0, 0 ,0],
    [1, 1, 1, 1],
    [0, 1, 1, 2]
]

df_predicao = pd.DataFrame(novos_registros, columns=[colunas_modelo])
df_predicao['classe_predita'] = model.predict(df_predicao.values)

print('Resultado da análçise dos novos registros com base no modelo treinado:')
print(df_predicao)