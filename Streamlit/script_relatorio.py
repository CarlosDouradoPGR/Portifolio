import pandas as pd
import numpy as np
import streamlit as st

# Visualização
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Criando o modelo
renda = pd.read_csv("input/previsao_de_renda.csv")

# Categorizando tempo_emprego
renda["cat_tempo_emprego"] = None
renda.loc[renda["tempo_emprego"] <= 1, "cat_tempo_emprego"] = "1 ano ou menos"
renda.loc[(renda["tempo_emprego"] > 1) & (renda["tempo_emprego"] <= 5), "cat_tempo_emprego"] = "entre 5 e 1"
renda.loc[(renda["tempo_emprego"] > 5) & (renda["tempo_emprego"] <= 10), "cat_tempo_emprego"] = "entre 5 e 10"
renda.loc[renda["tempo_emprego"] > 10, "cat_tempo_emprego"] = "mais que 10"

# Transformando data_ref para datetime
renda["data_ref"] = pd.to_datetime(renda["data_ref"])

# Retirando dados nulos
renda.dropna(inplace=True)

# Criando uma variável log_renda para verificar o desenho dos pontos
renda["log_renda"] = np.log(renda["renda"])

# Criando dumies para as variáveis categóricas.
renda["viuvo_T"] = renda["estado_civil"].apply(lambda x: 1 if x == "Viúvo" else 0)
renda["pensionista_T"] = renda["tipo_renda"].apply(lambda x: 1 if x == "Pensionista" else 0)
renda["superior_T"] = renda["educacao"].apply(lambda x: 1 if x == "Superior completo" else 0 )
renda["estudio_T"] = renda["tipo_residencia"].apply(lambda x: 1 if x == "Estúdio" else 0 )
renda["sexo_M"] = pd.get_dummies(renda["sexo"], drop_first=True)


# -------------------------------- Criando o modelo---------------------------------------------------------------------
# Variável explicativa
X = renda[['posse_de_veiculo','idade', 'tempo_emprego', 'qt_pessoas_residencia','sexo_M', "qtd_filhos"]]

# Variável resposta
y = renda["renda"]

# Separando em base treino e base teste
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=402, train_size=0.3)



# Ajustando a variável resposta para logartimo para evitar o overfiting
y_train = np.log(y_train)

# Melhor árvore de regressão
clf_reg = DecisionTreeRegressor(random_state=4002, ccp_alpha=0.006)

clf_reg.fit(X_train, y_train)

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Inicando o relatório-------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

df = pd.read_csv("clientes_novos.csv")


def relatorio(novos:pd.DataFrame):

    pred = pd.DataFrame({"renda" : np.exp(clf_reg.predict(novos))})
    pred["emprestimo"] = pred["renda"].apply(lambda x: round (x*10, 2))
    taxa = ((2.15 / 100) + 1) ** 36

    novos["renda_predita"] = pred["renda"].apply(lambda x: round(x, 2))
    novos["valor_liberado"] = pred["emprestimo"]
    novos["valor_final"] = pred["emprestimo"].apply(lambda x: round(x * taxa, 2))
    novos["juros"] = pred["emprestimo"].apply(lambda x: round(x * taxa - x, 2))
    novos["parcela"] = novos["valor_final"].apply(lambda x: round(x / 36, 2))

    return novos


# print(relatorio(df)[["renda_predita", "valor_liberado", "valor_final", "parcela", "juros"]])


# ------------------------------------Streamlit------------------------------------------------------------------------


st.markdown("# Relatório")
st.markdown("Nesse relátorio pode ser vizualisado o resultado do modelo criado para a previsão de renda e aprovações do"
            "empréstimo da empresa.")
st.markdown("### Base de dados dos novos clientes")
st.markdown("Essa é a base de dados dos novos clientes, onde será aplicado o modelo de Regressão Linear, para predizer"
            "uma possivel renda e estimular o valor do empréstimo do cliente")
st.write(df)
st.markdown('Nesse histogrma verificamos a normalidade dos dados, como podemos ver, a idade dos clientes segue uma '
            'normalidade, diferente do tempo de trabalho, onde a maioria dos clientes estão ente 0 e 10 no quesito tempo'
            'de emprego.')
fig1, axes = plt.subplots(1,2, figsize=[10,4])
sns.histplot(ax=axes[0], x=df["idade"], kde=True, bins=40)
axes[0].set_title("Histograma idade")
axes[0].set_ylabel("Frequência")
sns.histplot(ax=axes[1] ,x=df["tempo_emprego"], kde=True, bins=40)
axes[1].set_title("Histograma tempo_emprego")
axes[1].set_ylabel("Frequência")

st.pyplot(fig1)

st.markdown("### Base de dados dos empréstimos solicitados")

st.markdown("Após a aplicação do modelo nos novos clientes, obtemos essa tabela, onde é adicionado o valor da renda "
            "predita pelo modelo, valor do liberado para  (10 x  a renda), o valor total que o cliente vai pagar, o "
            "juros e a parcela mensal")

st.write(relatorio(df))

st.markdown("Abaixo temos o gráfico de boxblot, que nos ajuda a entender que a maioria dos valores liberados estão entre"
            "25000 e 100000, contendo apenas dois valores fora desse scopo ")

fig3, axes3 = plt.subplots(figsize=[10,3])
sns.boxplot(x=df["valor_liberado"])
st.pyplot(fig3)


st.markdown("#### Análise de Retorno de Investimento")
st.markdown("Aqui temos uma visualização do retorno do empréstimo considerando que todas as parcelas sejam pagas em dias"
            "e sem acrécimo de juros.")

evasao = pd.DataFrame()

evasao["mes"] = [i for i in range(0, 38, 1)]
evasao["caixa"] = None

caixa = list()
mes0 =  - df["valor_liberado"].sum()
parcela = df["parcela"].sum()
caixa.append(mes0)




for i in range(0, 37, 1):
    mes0 += parcela
    caixa.append(round(mes0, 2))

evasao["caixa"] = caixa

plt.close("all")
fig2, axes2 = plt.subplots(figsize=[10,4])
plt.axhline(y=0, color="r", linestyle="--")
sns.lineplot(x="mes", y="caixa", data=evasao)



st.pyplot(fig2)








#st.write(df)