import streamlit as st
import requests
import time

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Text Summarization App",
    layout="centered"
)

st.markdown("# Text Summarization App")
st.write("Enter your text below and generate a concise summary.")

text_input = st.text_area(
    label="Enter your text",
    height=250,
    placeholder="Enter an article you want to summarize..."
)

if st.button("Submit"):
    if not text_input.strip():
        st.warning("Please enter some text before submitting.")
    else:
        with st.spinner("Summarizing..."):
            try:
                start_time = time.time()

                response = requests.post(
                    API_URL,
                    json={"text": text_input},
                    timeout=60
                )

                latency = round(time.time() - start_time, 2)

                if response.status_code == 200:
                    summary = response.json()["summary"]

                    st.subheader("Summary")
                    st.write(summary)
                    st.caption(f"Inference time: {latency} seconds")

                else:
                    error_msg = response.json().get("detail", "Unknown error")
                    st.error(error_msg)

            except requests.exceptions.RequestException:
                st.error("Could not connect to FastAPI backend.")
