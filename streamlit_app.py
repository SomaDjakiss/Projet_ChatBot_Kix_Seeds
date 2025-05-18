import os
import re
import pandas as pd
import streamlit as st
from collections import defaultdict
from itertools import islice
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI


st.set_page_config(page_title="🎓 Analyse Scolaire", layout="centered")
st.title("🎓 Chatbot Scolaire - Analyse des Performances")

# ✅ Chargement et cache des données
df=pd.read_csv("https://raw.githubusercontent.com/SomaDjakiss/Projet_ChatBot_Kix_Seeds/main/data_kix_seeds.csv",encoding="ISO-8859-1",sep=";")
# ✅ Chargement de la clé API OpenAI depuis les secrets Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Configuration du modèle GPT
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "donnees"],
    template="""
Tu es un expert en analyse pédagogique,  conçue pour fournir des réponses précises, structurées et basées sur des données scolaires.

Voici des données sur les performances scolaires d'élèves d'une même classe. Chaque bloc correspond à un élève.

Ta tâche est :
### question concernant un élève :
**Pour un élève spécifique** (par nom ou ID) :
- Affiche ses notes : notes_matieres, moyenne_t1, moyenne_t2, moyenne_t3.

- Donne son rang : rang_t1, rang_t2, rang_t3.

- Résume ses absences : type_presence, motif_absence.

- Analyse ses forces (matières avec de bonnes notes) et ses faiblesses (matières avec des notes faibles).

- Identifie les tendances générales : progrès, régressions, constance.

- Compare sa performance à celle de sa classe.

- Fournis un bilan global et par matière de ses résultats.

 - Propose des suggestions personnalisées et des conseils pédagogiques concrets pour améliorer ses performances.

### question concernant un classe:
- Fournis la moyenne générale de la classe (moyenne_classe_t1, moyenne_classe_t2, moyenne_classe_t3)

** identifie:
- Le meilleur élève et le plus faible par trimestre.

- Les matières les plus réussies et les plus échouées.

- Fournis les statistiques descriptives : moyennes, maximums, effectifs.

- Repère les tendances générales :

- élèves performants, moyens, faibles.

- éventuels cas exceptionnels (très bons ou très faibles).

- Analyse trois élèves représentatifs de cette classe.

- Donne un aperçu des écarts de performance.

- Propose des pistes pédagogiques concrètes pour renforcer les acquis ou combler les lacunes.

### question concernan une école:
**Dresse un bilan *par classe* :
- Moyenne générale de chaque classe.
- faire analyse sur chaque école en précisant la moyenne générale de la nom_salle_classe(CP1,CP2,CE1,CE2,CM1 et CM2) dans une école données
- Matières les plus réussies / échouées.
** Intègre aussi :
-  Les cas de *violence ou de victimisation* s'ils sont signalés.
- Les caractéristiques spécifiques de l'école (environnement, effectif, encadrement, etc.).
- Suggère des recommandations réalistes pour améliorer la qualité de l'enseignement dans l'établissement.

###Si la question concerne une CEB ou une commune
**Présente une *analyse comparative entre écoles* :
- Performances globales par classe et par école.
- Classement ou hiérarchisation des écoles si pertinent.
- Forces et faiblesses communes ou spécifiques.
- Signalement des situations problématiques (violences, inégalités, déséquilibres).
- Propose des recommandations *à l'échelle territoriale* (CEB ou commune) pour renforcer l'apprentissage et réduire les disparités.

###Objectif final 
**Fournir une *analyse claire, structurée et compréhensible*, avec :
- Des *constats basés sur les données*.
- Des *conclusions pédagogiques* pertinentes.
- Des *recommandations pratiques* pour améliorer les performances à tous les niveaux analysés.

**Ne jamais inventer de données**. Si les données sont manquantes, indique-le clairement.
repond correctement les questions en utilisant des phrases claire et professionelle


Question : {question}

Données :
{donnees}

Fais une réponse claire et structurée.
"""
)

def extraire_filtre(question, valeurs_connues):
    for val in valeurs_connues:
        if val and str(val).lower() in question.lower():
            return val
    return None

def get_response_from_dataframe(question, df, nb_eleves=3):
    reponses = []

    # Normaliser les colonnes pour les correspondances
    question_lower = question.lower()

    # Recherche des filtres possibles
    id_eleve = extraire_filtre(question_lower, df['id_eleve'].astype(str).unique())
    identifiant_unique = extraire_filtre(question_lower, df['identifiant_unique_eleve'].astype(str).unique())
    id_classe = extraire_filtre(question_lower, df['id_classe'].astype(str).unique())
    code_classe = extraire_filtre(question_lower, df['code_classe'].astype(str).unique())
    nom_classe = extraire_filtre(question_lower, df['nom_classe'].astype(str).unique())
    nom_ecole = extraire_filtre(question_lower, df['nom_ecole'].astype(str).unique())
    code_ecole = extraire_filtre(question_lower, df['code_ecole'].astype(str).unique())
    ceb = extraire_filtre(question_lower, df['ceb_ecole'].astype(str).unique())
    commune = extraire_filtre(question_lower, df['commune_ecole'].astype(str).unique())
    ecole_id = extraire_filtre(question_lower, df['ecole_id'].astype(str).unique())

    # 🔍 Si recherche par élève
    if id_eleve or identifiant_unique:
        ident = id_eleve or identifiant_unique
        ligne = df[(df['id_eleve'].astype(str) == ident) | (df['identifiant_unique_eleve'].astype(str) == ident)]
        if not ligne.empty:
            ligne = ligne.iloc[0]
            donnees_texte = "\n".join([f"{col} : {ligne[col]}" for col in df.columns if col in ligne])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            if hasattr(resultat, 'content'):
                resultat = resultat.content
            return resultat

    # 🔍 Sinon : filtrage par classe et école
    filtres = []
    if nom_ecole: filtres.append(df['nom_ecole'].str.lower() == nom_ecole.lower())
    if code_ecole: filtres.append(df['code_ecole'].astype(str) == str(code_ecole))
    #if ceb: filtres.append(df['ceb_ecole'].str.lower() == ceb.lower())
    if ceb: filtres.append(df['ceb_ecole'].astype(str) == str(ceb))
    #if commune: filtres.append(df['commune_ecole'].str.lower() == commune.lower())
    if commune: filtres.append(df['commune_ecole'].astype(str) == str(commune))
    if code_classe: filtres.append(df['code_classe'].astype(str) == str(code_classe))
    if nom_classe: filtres.append(df['nom_classe'].str.lower() == nom_classe.lower())
    if id_classe: filtres.append(df['id_classe'].astype(str) == str(id_classe))
    if ecole_id: filtres.append(df['ecole_id'].astype(str) == str(ecole_id))

    if filtres:
        from functools import reduce
        import operator
        condition = reduce(operator.and_, filtres)
        df_filtre = df[condition]
        if df_filtre.empty:
            return "Aucune donnée trouvée avec les critères spécifiés."

        # Limiter le nombre d'élèves analysés
        df_limite = df_filtre.head(nb_eleves)
        for _, ligne in df_limite.iterrows():
            donnees_texte = "\n".join([f"{col} : {ligne[col]}" for col in df.columns if col in ligne])
            prompt = prompt_template.format(question=question, donnees=donnees_texte)
            resultat = llm.invoke(prompt)
            if hasattr(resultat, 'content'):
                resultat = resultat.content
            reponses.append(str(resultat))
        return "\n\n---\n\n".join(reponses)

    return "Aucun filtre détecté dans la question. Veuillez spécifier un élève, une classe ou une école."

# Interface Streamlit
#st.set_page_config(page_title="🎓 Analyse Scolaire", layout="centered")
#st.title("🎓 Chatbot Scolaire - Analyse des Performances")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#  Formulaire avec champ texte et bouton Submit
with st.form("formulaire_question"):
    user_input = st.text_input("Pose ta question sur un élève, une école ou une classe")
    submitted = st.form_submit_button("Submit")

#  Traitement après envoi
if submitted and user_input:
    response = get_response_from_dataframe(user_input, df)
    st.session_state.chat_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ])

# Affichage de l’historique
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
