# ==========================
#   IMPORTS
# ==========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
#   CONFIG STREAMLIT
# ==========================
st.set_page_config(
    page_title="Prédiction Prix Maison",
    page_icon="🏠",
    layout="centered"
)


# ==========================
#   CHARGEMENT DONNÉES
# ==========================
df = pd.read_csv("kc_house_data1.csv")  

X = df.drop("price", axis= 1)
y = df["price"]

# ==========================
#   SPLIT DONNÉES
# ==========================
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

# ==========================
#   MODÈLE RL
# ==========================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================
#   PERFORMANCE
# ==========================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.sidebar.write("📈 Réel vs Prédit")

fig1, ax1 = plt.subplots(figsize=(4, 4))
ax1.scatter(y_test, y_pred)
ax1.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax1.set_xlabel("Réel")
ax1.set_ylabel("Prédit")
ax1.set_title("Réel vs Prédit")

st.sidebar.pyplot(fig1)


# ==========================
#   INTERFACE UTILISATEUR
# ==========================
st.title("🏠 Prédiction du Prix d'une Maison")
st.write("Veuillez renseigner les caractéristiques du logement :")

col1, col2 = st.columns(2)

with col1:
    bathrooms = st.number_input(
        "Nombre de salles de bain",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.25
    )

    sqft_living = st.number_input(
        "Surface habitable (sqft)",
        min_value=300,
        max_value=10000,
        value=1500
    )

    grade = st.slider(
        "Qualité de construction (grade)",
        min_value=1,
        max_value=13,
        value=7
    )

with col2:
    sqft_above = st.number_input(
        "Surface hors sous-sol (sqft_above)",
        min_value=300,
        max_value=10000,
        value=1200
    )

    sqft_living15 = st.number_input(
        "Surface moyenne du voisinage (sqft_living15)",
        min_value=300,
        max_value=10000,
        value=1500
    )

# ==========================
#   PRÉDICTION
# ==========================
if st.button("🔮 Prédire le prix"):
    input_data = np.array([[
        bathrooms,
        sqft_living,
        grade,
        sqft_above,
        sqft_living15
    ]])

    prediction = model.predict(input_data)
    prix = float(prediction[0])

    st.success(f"💰 Prix estimé de la maison : **{prix:,.2f} $**")

# ==========================
#   PERFORMANCE MODÈLE
# ==========================
st.subheader("📊 Performance du modèle (Régression Linéaire)")

col3, col4, col5 = st.columns(3)

col3.metric("R² Score", f"{r2:.3f}")
col4.metric("MAE ($)", f"{mae:,.0f}")
col5.metric("RMSE ($)", f"{rmse:,.0f}")

# ==========================
#   EXPLICATION
# ==========================
st.info(
    "Ce modèle utilise une **Régression Linéaire** pour estimer le prix "
    "d’une maison à partir de ses caractéristiques principales. "
    "Les métriques affichées permettent d’évaluer la qualité des prédictions."
)
