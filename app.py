import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

@st.cache_resource
def load_and_train():
    from datasets import load_dataset
    dataset = load_dataset("stanfordnlp/imdb")
    
    train_df = pd.DataFrame(dataset['train']).sample(10000, random_state=42)
    test_df  = pd.DataFrame(dataset['test']).sample(10000, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(train_df['text'])
    X_test_vec  = vectorizer.transform(test_df['text'])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, train_df['label'])

    acc = accuracy_score(test_df['label'], model.predict(X_test_vec))
    return model, vectorizer, acc

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠")
st.title("🧠 AI Sentiment Analyzer")
st.write("Built using Machine Learning — Logistic Regression + TF-IDF")

model, vectorizer, acc = load_and_train()
st.success(f"✅ Model trained! Accuracy: {acc*100:.1f}%")

st.subheader("Analyze any review or sentence:")
user_input = st.text_area("Type or paste text here:", height=150,
                           placeholder="e.g. The movie was absolutely fantastic!")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        vec   = vectorizer.transform([user_input])
        pred  = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        label = "🟢 POSITIVE" if pred == 1 else "🔴 NEGATIVE"
        conf  = proba[pred] * 100

        st.markdown(f"### Result: {label}")
        st.markdown(f"**Confidence:** {conf:.1f}%")

        fig, ax = plt.subplots(figsize=(5, 2.5))
        bars = ax.barh(["Negative", "Positive"], [proba[0]*100, proba[1]*100],
                       color=["#ef4444", "#22c55e"])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Prediction Confidence")
        for bar, val in zip(bars, [proba[0]*100, proba[1]*100]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va='center', fontsize=11)
        st.pyplot(fig)

st.markdown("---")
st.subheader("Try these examples:")
examples = [
    "This film was a masterpiece. I was blown away by the acting!",
    "Terrible movie. Complete waste of time and money.",
    "It was okay, nothing special but not bad either."
]
for ex in examples:
    if st.button(ex[:60] + "..."):
        st.info(f"Copy this into the box above: \n\n *{ex}*")