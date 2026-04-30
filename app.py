import streamlit as st
from transformers import pipeline

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    ner_pipeline = pipeline(
        "ner",
        model="./NER",
        tokenizer="./NER",
        aggregation_strategy=None   # we handle combining manually
    )
    return ner_pipeline

ner = load_model()

# -------------------------------
# Combine WordPiece Tokens
# -------------------------------
def combine_wordpieces(entities):
    combined = []
    current_word = ""
    current_label = None
    current_score = []

    for ent in entities:
        word = ent['word']
        label = ent['entity']
        score = ent['score']

        if word.startswith("##"):
            current_word += word[2:]
            current_score.append(score)
        else:
            if current_word:
                combined.append({
                    "word": current_word,
                    "entity": current_label,
                    "score": sum(current_score) / len(current_score)
                })
            current_word = word
            current_label = label
            current_score = [score]

    if current_word:
        combined.append({
            "word": current_word,
            "entity": current_label,
            "score": sum(current_score) / len(current_score)
        })

    return combined

# -------------------------------
# Format Labels (User-Friendly)
# -------------------------------
def format_label(label):
    clean_label = label.replace("B-", "").replace("I-", "")
    
    label_map = {
        "PER": "Person",
        "ORG": "Organization",
        "LOC": "Location",
        "MISC": "Miscellaneous"
    }
    
    return label_map.get(clean_label, clean_label)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Named Entity Recognition App", layout="centered")

st.title("🧠 Named Entity Recognition")
st.write("Enter text and identify entities like **Person, Organization, Location**.")

text_input = st.text_area("Enter your text here:", height=150)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Recognize Entities"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        results = ner(text_input)

        # Combine tokens
        results = combine_wordpieces(results)

        if len(results) == 0:
            st.info("No entities found.")
        else:
            st.subheader("📌 Detected Entities")

            for entity in results:
                word = entity["word"]
                label = format_label(entity["entity"])
                score = entity["score"]

                st.markdown(
                    f"""
                    ### 🔹 {word}  
                    **Type:** {label}  
                    **Confidence:** {score:.2f}
                    """
                )