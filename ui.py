import streamlit as st
import time

from src.summarizer.pipeline.prediction import PredictionPipeline


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Text Summarization App",
    layout="centered",
)

st.title("Text Summarization App")
st.write(
    "Paste a long article below and generate a concise summary using a Transformer model."
)


# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_predictor():
    """
    Loads the model + tokenizer once per app session.
    Streamlit Cloud friendly.
    """
    return PredictionPipeline()


try:
    predictor = load_predictor()
except Exception as e:
    st.error("Failed to load the model.")
    st.exception(e)
    st.stop()


# -----------------------------
# User input
# -----------------------------
text_input = st.text_area(
    label="Enter text to summarize",
    height=280,
    placeholder="Paste an article or long text here...",
)


# -----------------------------
# Inference
# -----------------------------
if st.button("Summarize"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text before clicking Summarize.")
    else:
        with st.spinner("Generating summary..."):
            try:
                start_time = time.time()
                summary = predictor.predict(text_input)
                latency = round(time.time() - start_time, 2)

                st.subheader("Summary")
                st.write(summary)
                st.caption(f"Inference time: {latency} seconds")

            except ValueError as ve:
                # Token length or validation errors
                st.error(str(ve))

            except Exception as e:
                st.error(" An unexpected error occurred during inference.")
                st.exception(e)



st.markdown("---")
st.caption(
    "Built with 🤗 Transformers & Streamlit | "
    "End-to-end NLP project for real-world deployment"
)
