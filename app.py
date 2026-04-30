import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    ner_pipeline = pipeline(
        "ner",
        model="./NER",
        tokenizer="./NER",
        aggregation_strategy=None   # manual handling
    )
    return ner_pipeline

ner = load_model()

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

def detailed_label(label):
    label_map = {
        "B-PER": "Person (Start of Name)",
        "I-PER": "Person (Continuation of Name)",
        "B-ORG": "Organization (Start)",
        "I-ORG": "Organization (Continuation)",
        "B-LOC": "Location (Start)",
        "I-LOC": "Location (Continuation)",
        "B-MISC": "Miscellaneous (Start)",
        "I-MISC": "Miscellaneous (Continuation)"
    }
    return label_map.get(label, label)

st.set_page_config(page_title="Named Entity Recognition App", layout="centered")

st.title("Named Entity Recognition")
st.write("Enter text and identify entities with detailed explanations.")

text_input = st.text_area("Enter your text here:", height=150)

if st.button("Recognize Entities"):
    if text_input.strip() == "":
        st.warning("Please enter some text")
    else:
        results = ner(text_input)

        combined_results = combine_wordpieces(results)

        if len(results) == 0:
            st.info("No entities found.")
        else:
            st.subheader("Detected Entities")

            for i, entity in enumerate(combined_results):
                word = entity["word"]
                label = detailed_label(entity["entity"])
                score = entity["score"]

                st.markdown(
                    f"""
                    ### 🔹 {word}
                    **Type:** {label}  
                    **Confidence:** {score:.2f}
                    """
                )

            with st.expander("Show Token-Level Details"):
                for ent in results:
                    st.write(
                        f"{ent['word']} → {detailed_label(ent['entity'])} "
                        f"(Score: {ent['score']:.2f})"
                    )