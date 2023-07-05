"""Prompt Evaluator."""

import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

HF_MODEL = "tiiuae/falcon-7b-instruct"
HF_DATASET = "amazon_polarity"

SEED = 42
NUM_SAMPLES = 25
PROMPT_TEMPLATE_HEIGHT = 200

TEXT_COLUMN = "content"
ANNOTATION_COLUMN = "label"

STARTER_PROMPT = """Classify whether the following sentence has a positive or negative sentiment.

Sentence: ```{text}```

Sentiment [Positive/Negative]: """


def prepare_dataset():
    dataset_df = (
        load_dataset(HF_DATASET, split="test")
        .shuffle(seed=SEED)
        .select(range(NUM_SAMPLES))
        .to_pandas()
    )

    dataset_df["label"].replace({1: "Positive", 0: "Negative"}, inplace=True)

    return dataset_df


def infer(prompt_template, text, model):
    prompt = prompt_template.format(text=text)
    response = st.session_state.client.post(json={"inputs": prompt}, model=model)
    output = response.json()[0]["generated_text"]
    output = output if "https" in model else output[len(prompt) :]
    # output = st.session_state.client.conversational(prompt, model=model)
    return output


def infer_multi(prompt_template, text_series, model, progress=None):
    props = (i / len(text_series) for i in range(1, len(text_series) + 1))

    def infer_with_progress(text):
        output = infer(prompt_template, text, model)
        if progress is not None:
            progress.progress(next(props))
        return output.strip()

    return text_series.apply(infer_with_progress)


def measure(dataset, infereces):
    evaluation_df = pd.DataFrame(
        {
            "hit/miss": np.where(
                dataset[ANNOTATION_COLUMN] == infereces, "hit", "miss"
            ),
            "annotation": dataset[ANNOTATION_COLUMN],
            "inferece": infereces,
            "text": dataset[TEXT_COLUMN],
        }
    )

    annotation_labels = sorted(evaluation_df["annotation"].unique())
    inference_labels = sorted(evaluation_df["inferece"].unique())
    all_labels = sorted(set(annotation_labels + inference_labels))

    acc = accuracy_score(evaluation_df["annotation"], evaluation_df["inferece"])
    cm = confusion_matrix(
        evaluation_df["annotation"], evaluation_df["inferece"], labels=all_labels
    )

    cm_display = ConfusionMatrixDisplay(cm, display_labels=all_labels).plot().figure_

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "confusion_matrix_display": cm_display,
        "hit_miss": evaluation_df,
        "annotation_labels": annotation_labels,
        "inference_labels": inference_labels,
    }

    return metrics


def run_evaluation(prompt_template, dataset, model, progress=None):
    inferences = infer_multi(prompt_template, dataset[TEXT_COLUMN], model, progress)
    metrics = measure(dataset, inferences)
    return metrics


def combine_labels(labels):
    return " | ".join(f"`{label}`" for label in labels)


if "dataset" not in st.session_state:
    st.session_state["dataset"] = prepare_dataset()
if "client" not in st.session_state:
    st.session_state["client"] = InferenceClient(token=st.secrets.get("hf_token"))

st.title("Prompt Evaluator")

with st.form("prompt_template_form"):
    model = st.text_input("Model", HF_MODEL)
    prompt_template = st.text_area(
        "Prompt Template", STARTER_PROMPT, height=PROMPT_TEMPLATE_HEIGHT
    )
    submitted = st.form_submit_button("Evaluate")

    if submitted:
        if not model:
            st.error("Model must be specified.")
            st.stop()
        if not prompt_template:
            st.error("Prompt template must be specified.")
            st.stop()

        inference_progress = st.progress(0, "Executing Inference")

        try:
            evaluation = run_evaluation(
                prompt_template, st.session_state.dataset, model, inference_progress
            )
        except HfHubHTTPError as e:
            st.error(e)
            st.stop()

        st.markdown(
            f"Annotation labels: {combine_labels(evaluation['annotation_labels'])}"
        )
        st.markdown(
            f"Inference labels: {combine_labels(evaluation['inference_labels'])}"
        )

        st.metric("Accuracy", evaluation["accuracy"])

        st.markdown("## Confusion Matrix")
        st.pyplot(evaluation["confusion_matrix_display"])

        st.markdown("## Hits and Misses")
        st.dataframe(evaluation["hit_miss"])
