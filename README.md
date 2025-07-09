# Iris Species Classifier 🌸

An interactive web app built with [Streamlit](https://streamlit.io/) that demonstrates machine learning classification using the classic Iris dataset and a Random Forest model. Users can explore how different flower measurements affect the predicted species in real time.

---

## Table of Contents
- [Overview](#overview)
- [What is Streamlit?](#what-is-streamlit)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Customization](#customization)
- [References](#references)

---

## Overview
This app allows users to:
- Adjust the four main measurements of an iris flower (sepal length, sepal width, petal length, petal width) using sliders.
- Instantly see the predicted species (Setosa, Versicolor, Virginica) based on a trained Random Forest model.
- View the prediction probabilities for each species in a bar chart.
- Preview random samples from the dataset.

---

## What is Streamlit?
[Streamlit](https://streamlit.io/) is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. With Streamlit, you can turn Python scripts into interactive apps in minutes, all with minimal code and no need for web development experience.

---

## Features
- **Interactive Sliders:** Adjust flower measurements and see predictions update live.
- **Models:** Uses basic ML models(scikit-learn) for robust predictions.
- **Dataset Preview:** Explore random samples from the Iris dataset.
- **Probability Visualization:** View prediction probabilities as a bar chart (Altair).
- **Model-Accuracy** View the applied Model's accuracy.
- **Fast & Cached:** Data loading and model training are cached for instant interaction.
- **Deploy-Ready:** Easily deploy to Streamlit Community Cloud or HuggingFace Spaces.

---

## How It Works
1. **Data Loading:** Loads the Iris dataset from scikit-learn and prepares it as a DataFrame.
2. **Model Training:** Trains the selected model on the dataset (caching for speed).
3. **User Input:** Users adjust sliders in the sidebar to set flower measurements.
4. **Prediction:** The model predicts the species and shows the probability for each class.
5. **Visualization:** Results are displayed as text and a bar chart.

---

## Installation

1. **Clone the repository or download the script:**
   ```bash
   git clone <https://github.com/cbbisht2004/CLASSIFICATION>
   ```

2. **Install dependencies:**
   It's recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run "Classification_Iris.py"
   ```

2. **Interact:**
   - Pick a model from the drop-down menu (Random Forest, SVM, Logistic Regression, KNN)
   - Use the sidebar sliders to adjust flower measurements.
   - View the predicted species and probability chart.
   - Expand the dataset preview to see random samples.

4. **Stop the app:**
   - Press `Ctrl+C` in the terminal.

---

## File Structure

```
CLASSIFICATION/
├── Classification_Iris.py                  # Main Streamlit app
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

---

## Customization
- **Model:** You can swap out the Random Forest for another classifier (e.g., SVM, Logistic Regression) in the code.
- **Features:** Add more visualizations or explanations as desired.
- **Deployment:** Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) or [HuggingFace Spaces](https://huggingface.co/spaces) for public access.

---

## References
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn: Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [UCI Machine Learning Repository: Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris)
- [Altair Documentation](https://altair-viz.github.io/)

---

**Author:** Chetan Bisht
