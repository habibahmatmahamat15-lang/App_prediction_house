# Importer les bibliothèques
import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Prédiction des Prix des Maisons", page_icon="🏠", layout="wide")


# Page d'accueil

    # Image de bienvenue (assurez-vous que 'image.jpg' est dans le même dossier ou commentez cette ligne si vous n’avez pas d’image)
st.image("image.jpg", caption="Bienvenue à la Prédiction des Prix des Maisons !", width=700)
    # Titre
st.title("Prédiction des Prix des Maisons")

    # Texte explicatif
st.markdown("""
        Ce modèle prédit le prix des maisons en se basant sur des caractéristiques telles que 
        le nombre moyen de pièces, la proportion de terrains résidentiels et d’autres facteurs.
    """)

    # Liens externes
st.markdown("### Consultez le modèle et le code :")
st.markdown("[Voir le code sur GitHub :](https://github.com/habibahmatmahamat15-lang/App_prediction_house.git)")
