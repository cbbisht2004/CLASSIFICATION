"""Streamlit Iris Species Classifier
-----------------------------------
An interactive demo that trains a RandomForest model on the classic
Iris dataset and lets users explore predictions by adjusting sliders.
Deploy-ready for Streamlit Community Cloud / HuggingFace Spaces.
"""

import streamlit as st
import pandas as pd
import altair as alt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------------
# Page configuration & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Data loading & model training helpers (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading Iris dataset...")
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target
    return df, iris.target_names

@st.cache_resource(show_spinner="Training RandomForest model...")
def train_model(df: pd.DataFrame, *, random_state: int = 42):
    X = df.iloc[:, :-1]
    y = df["species"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model

# ----------------------------------------------------------------------------
# Main app logic
# ----------------------------------------------------------------------------

df, target_names = load_data()
model = train_model(df)

# ----- Header -----
st.title("🌸 Iris Flower Species Classifier")

# ----- Dataset preview -----
with st.expander("🔎  Preview random samples from the dataset"):
    df_preview = df.sample(10, random_state=1).copy()
    df_preview["species"] = df_preview["species"].apply(lambda x: target_names[x])
    st.dataframe(df_preview, use_container_width=True)

# ----- Sidebar controls -----
st.sidebar.header("Adjust flower measurements (cm)")

feature_inputs = {
    feature: st.sidebar.slider(
        label=feature.replace(" (cm)", ""),
        min_value=float(df[feature].min()),
        max_value=float(df[feature].max()),
        value=float(df[feature].mean()),
    )
    for feature in df.columns[:-1]
}

# ----- Prediction -----
features = list(feature_inputs.values())
prediction_idx = int(model.predict([features])[0])
prediction_proba = model.predict_proba([features])[0]

st.subheader("🔮 Prediction")
st.write(
    f"Based on the parameters you chose, the model predicts this flower is "
    f"**{target_names[prediction_idx]}**."
)

# Probability chart
prob_df = pd.DataFrame(
    {
        "species": target_names,
        "probability": prediction_proba,
    }
)

bar_chart = (
    alt.Chart(prob_df)
    .mark_bar(color="#4e79a7", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
    .encode(
        x=alt.X("species", sort=None, title="Species"),
        y=alt.Y("probability", title="Prediction Probability", scale=alt.Scale(domain=[0, 1])),
        tooltip=["species", "probability"]
    )
    .properties(height=350)
    .configure_axis(labelFontSize=12, titleFontSize=14)
    .configure_view(stroke=None)
)

st.altair_chart(bar_chart, use_container_width=True)

# Final result in uppercase
st.success(f"🌼 Final Predicted Species:  {target_names[prediction_idx].upper()}")

# Footer
st.caption(
    "Built with Streamlit · Code and model cached for instant interaction · "
    "Data: UCI Iris (Fisher, 1936)"
)

