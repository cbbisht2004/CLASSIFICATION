"""Streamlit Iris Species Classifier – Model Selector
----------------------------------------------------
Interactively predict Iris flower species with different scikit‑learn
classifiers. Users can tweak feature sliders **and** switch between
models in the sidebar.  Now also shows each model’s test accuracy.
"""

import streamlit as st
import pandas as pd
import altair as alt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score          # ⬅ added

# ─────────────────────────────── Page config ────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────── Data‑loading helper (cached) ───────────────────────
@st.cache_data(show_spinner="Loading Iris dataset…")
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

# ─────────────────────── Model factory & trainer (cached) ────────────────────
MODEL_FACTORY = {
    "Random Forest":        lambda rs: RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
    "Logistic Regression":  lambda rs: LogisticRegression(max_iter=200, random_state=rs),
    "K‑Nearest Neighbors":  lambda rs: KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": lambda rs: SVC(probability=True, random_state=rs),
}

@st.cache_resource(show_spinner="Training model…")
def train_model(df: pd.DataFrame, model_name: str, *, random_state: int = 42):
    """Return (fitted model, test‑set accuracy)."""
    X = df.iloc[:, :-1]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = MODEL_FACTORY[model_name](random_state)
    model.fit(X_train, y_train)

    test_acc = accuracy_score(y_test, model.predict(X_test))
    return model, test_acc

# ───────────────────────────── Main app starts ───────────────────────────────
df, target_names = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Select model & features")
model_choice = st.sidebar.selectbox("Machine Learning model", list(MODEL_FACTORY.keys()))

feature_inputs = {
    feature: st.sidebar.slider(
        label=feature.replace(" (cm)", ""),
        min_value=float(df[feature].min()),
        max_value=float(df[feature].max()),
        value=float(df[feature].mean()),
    )
    for feature in df.columns[:-1]
}

# Train / retrieve the chosen model
model, test_acc = train_model(df, model_choice)

# ── Header & preview ─────────────────────────────────────────────────────────
st.title("🌸 Iris Flower Species Classifier")

with st.expander("🔎 Preview random samples from the dataset"):
    preview = df.sample(10, random_state=1).copy()
    preview["species"] = preview["species"].apply(lambda x: target_names[x])
    st.dataframe(preview, use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────────
features = list(feature_inputs.values())
prediction_idx = int(model.predict([features])[0])
prediction_proba = model.predict_proba([features])[0] if hasattr(model, "predict_proba") else None

st.subheader("🔮 Prediction")
st.write(f"Model used: **{model_choice}**")
st.success(f"🌼 Final Predicted Species → {target_names[prediction_idx].upper()}")

# Show accuracy
st.info(f"Test‑set accuracy for **{model_choice}**: {test_acc:.2%}")

# ── Probability chart ────────────────────────────────────────────────────────
if prediction_proba is not None:
    prob_df = pd.DataFrame({"species": target_names, "probability": prediction_proba})

    bar = (
        alt.Chart(prob_df)
        .mark_bar(color="#4e79a7", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("species", sort=None, title="Species"),
            y=alt.Y("probability", title="Prediction Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["species", alt.Tooltip("probability", format=".2f")],
        )
        .properties(height=350)
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_view(stroke=None)
    )
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("Selected model does not provide class probabilities.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.caption(
    "Built with Streamlit · Switch between models in the sidebar · "
    "Data: UCI Iris (Fisher, 1936)"
)
