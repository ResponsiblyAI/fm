"""Prompter."""

import logging
import string

import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from spacy.lang.en import English

LOGGER = logging.getLogger(__name__)

TITLE = "Prompter"

FALLBACK_HF_MODEL = "timdettmers/guanaco-33b-merged"
HF_MODEL = st.secrets.get("hf_model", FALLBACK_HF_MODEL)

HF_DATASET = "amazon_polarity"

DATASET_SHUFFLE_SEED = 42
NUM_SAMPLES = 25
PROMPT_TEXT_HEIGHT = 300

TEXT_COLUMN = "content"
ANNOTATION_COLUMN = "label"

UNKNOWN_LABEL = "Unknown"

SEARCH_ROW_DICT = {"First": 0, "Last": -1}

# TODO: Change start temperature to 0.0 when HF supports it
GENERATION_CONFIG_PARAMS = {
    "temperature": {
        "NAME": "Temperature",
        "START": 0.1,
        "END": 5.0,
        "DEFAULT": 1.0,
        "STEP": 0.1,
        "SAMPLING": True,
    },
    "top_k": {
        "NAME": "Top K",
        "START": 0,
        "END": 100,
        "DEFAULT": 0,
        "STEP": 10,
        "SAMPLING": True,
    },
    "top_p": {
        "NAME": "Top P",
        "START": 0.1,
        "END": 1.0,
        "DEFAULT": 1.0,
        "STEP": 0.1,
        "SAMPLING": True,
    },
    "max_new_tokens": {
        "NAME": "Max New Tokens",
        "START": 16,
        "END": 256,
        "DEFAULT": 16,
        "STEP": 16,
        "SAMPLING": False,
    },
    "do_sample": {
        "NAME": "Sampling",
        "DEFAULT": False,
    },
}

GENERATION_CONFIG_DEFAULTS = {
    key: value["DEFAULT"] for key, value in GENERATION_CONFIG_PARAMS.items()
}

STARTER_PROMPT = """{text}

The sentiment of the text is"""


def prepare_datasets():
    label_dict = {0: normalize("negative"), 1: normalize("positive")}

    def load(split):
        df = (
            load_dataset(HF_DATASET, split=split)
            .shuffle(seed=DATASET_SHUFFLE_SEED)
            .select(range(NUM_SAMPLES))
            .to_pandas()
        )

        df["label"].replace(label_dict, inplace=True)
        df.drop(columns=["title"], inplace=True)

        return df

    return (load(split) for split in ("train", "test"))


def complete(prompt, generation_config):
    if generation_config is None:
        generation_config = {}

    # Reference for decoding stratagies:
    # https://huggingface.co/docs/transformers/generation_strategies

    # `text_generation_interface`
    # Currenly supports only `greedy` amd `sampling` decoding strategies
    # Following , we add `do_sample` if any of the other
    # samling related parameters are set
    # https://github.com/huggingface/text-generation-inference/blob/e943a294bca239e26828732dd6ab5b6f95dadd0a/server/text_generation_server/utils/tokens.py#L46

    # `transformers`
    # According to experimentations, it seems that `transformers` behave similarly

    # I'm not sure what is the right behavior here, but it is better to be explicit

    for name, params in GENERATION_CONFIG_PARAMS.items():
        # Checking for START to examine the a slider parameters only
        if (
            "START" in params
            and params["SAMPLING"]
            and name in generation_config
            and generation_config[name] is not None
        ):
            if generation_config[name] == params["DEFAULT"]:
                generation_config[name] = None
            else:
                assert generation_config["do_sample"]

    LOGGER.warning(f"API Call {generation_config=}")
    response = st.session_state.client.text_generation(
        prompt, stream=False, details=True, **generation_config
    )
    LOGGER.debug(response)

    output = response.generated_text

    # response = st.session_state.client.post(json={"inputs": prompt})
    # output = response.json()[0]["generated_text"]
    # output = st.session_state.client.conversational(prompt, model=model)
    # output = output if "https" in st.session_state.client.model else output[len(prompt) :]
    return output


def infer(prompt_template, text, generation_config=None):
    prompt = prompt_template.format(text=text)
    output = complete(prompt, generation_config)
    return output


def infer_multi(prompt_template, text_series, generation_config=None, progress=None):
    props = (i / len(text_series) for i in range(1, len(text_series) + 1))

    def infer_with_progress(text):
        output = infer(prompt_template, text, generation_config)
        if progress is not None:
            progress.progress(next(props))
        return output

    return text_series.apply(infer_with_progress)


def normalize(text):
    return text.strip().lower().capitalize()


def preprocess_output_line(text):
    return [normalize(str(token)) for token in st.session_state.tokenizer(text)]


# Inspired by OpenAI depcriated classification endpoint API
# https://github.com/openai/openai-cookbook/blob/main/transition_guides_for_deprecated_API_endpoints/classification_functionality_example.py
# https://help.openai.com/en/articles/6272941-classifications-transition-guide#h_e63b71a5c8
# Here we take the label from either the *first* or *last* (for CoT) line of the output
# This is not very robust, but it's a start that doesn't requires asking for a structured output such as JSON
def canonize_label(output, annotation_labels, search_row):
    assert search_row in SEARCH_ROW_DICT.keys()

    search_row_index = SEARCH_ROW_DICT[search_row]

    annotation_labels_set = set(annotation_labels)

    output_lines = output.strip("\n").split("\n")
    output_search_words = preprocess_output_line(output_lines[search_row_index])

    label_matches = set(output_search_words) & annotation_labels_set

    if len(label_matches) == 1:
        return next(iter(label_matches))
    else:
        return UNKNOWN_LABEL


def measure(dataset, outputs, search_row):
    annotation_labels = sorted(dataset[ANNOTATION_COLUMN].unique())

    inferences = [
        canonize_label(output, annotation_labels, search_row) for output in outputs
    ]

    inference_labels = annotation_labels.copy() + [UNKNOWN_LABEL]

    evaluation_df = pd.DataFrame(
        {
            "hit/miss": np.where(
                dataset[ANNOTATION_COLUMN] == inferences, "hit", "miss"
            ),
            "annotation": dataset[ANNOTATION_COLUMN],
            "inference": inferences,
            "output": outputs,
            "text": dataset[TEXT_COLUMN],
        }
    )

    all_labels = sorted(set(annotation_labels + inference_labels))

    acc = accuracy_score(evaluation_df["annotation"], evaluation_df["inference"])
    cm = confusion_matrix(
        evaluation_df["annotation"], evaluation_df["inference"], labels=all_labels
    )

    cm_display = ConfusionMatrixDisplay(cm, display_labels=all_labels)
    cm_display.plot()
    cm_display.ax_.set_xlabel("inference Labels")
    cm_display.ax_.set_ylabel("Annotation Labels")
    cm_display.figure_.autofmt_xdate(rotation=45)

    metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "confusion_matrix_display": cm_display.figure_,
        "hit_miss": evaluation_df,
        "annotation_labels": annotation_labels,
        "inference_labels": inference_labels,
    }

    return metrics


def run_evaluation(
    prompt_template, dataset, search_row, generation_config=None, progress=None
):
    outputs = infer_multi(
        prompt_template, dataset[TEXT_COLUMN], generation_config, progress
    )
    metrics = measure(dataset, outputs, search_row)
    return metrics


def combine_labels(labels):
    return "|".join(f"``{label}``" for label in labels)


if "client" not in st.session_state:
    st.session_state["client"] = InferenceClient(
        token=st.secrets.get("hf_token"), model=HF_MODEL
    )

if "tokenizer" not in st.session_state:
    st.session_state["tokenizer"] = English().tokenizer

if "train_dataset" not in st.session_state or "test_dataset" not in st.session_state:
    (
        st.session_state["train_dataset"],
        st.session_state["test_dataset"],
    ) = prepare_datasets()

if "generation_config" not in st.session_state:
    st.session_state["generation_config"] = GENERATION_CONFIG_DEFAULTS

st.set_page_config(page_title=TITLE, initial_sidebar_state="collapsed")

st.title(TITLE)

with st.sidebar:
    with st.form("model_form"):
        model = st.text_input("Model", HF_MODEL)

        # Defautlt values from:
        # https://huggingface.co/docs/transformers/v4.30.0/main_classes/text_generation
        # Edges values from:
        # https://docs.cohere.com/reference/generate
        # https://platform.openai.com/playground

        generation_config_sliders = {
            name: st.slider(
                params["NAME"],
                params["START"],
                params["END"],
                params["DEFAULT"],
                params["STEP"],
            )
            for name, params in GENERATION_CONFIG_PARAMS.items()
            if "START" in params
        }
        generation_config_checkbox = {
            name: st.checkbox(params["NAME"], params["DEFAULT"])
            for name, params in GENERATION_CONFIG_PARAMS.items()
            if "START" not in params
        }

        generation_config = generation_config_sliders | generation_config_checkbox

        seed = st.text_input("Seed").strip()

        submitted = st.form_submit_button("Set")

        if submitted:
            if not model:
                st.error("Model must be specified.")
                st.stop()

            if not seed:
                seed = None
            elif seed.isnumeric():
                seed = int(seed)
            else:
                st.error("Seed must be numeric or empty.")
                st.stop()

            generation_confing_slider_sampling = {
                name: value
                for name, value in generation_config_sliders.items()
                if GENERATION_CONFIG_PARAMS[name]["SAMPLING"]
            }
            if (
                any(
                    value != GENERATION_CONFIG_DEFAULTS[name]
                    for name, value in generation_confing_slider_sampling.items()
                )
                and not generation_config["do_sample"]
            ):
                sampling_slider_default_values_info = " | ".join(
                    f"{name}={GENERATION_CONFIG_DEFAULTS[name]}"
                    for name in generation_confing_slider_sampling
                )
                st.error(
                    f"Sampling must be enabled to use non default values for generation parameters: {sampling_slider_default_values_info}"
                )
                st.stop()

            if seed is not None and not generation_config["do_sample"]:
                st.error(
                    "Sampling must be enabled to use a seed. Otherwise, the seed field should be empty."
                )
                st.stop()

            generation_config["seed"] = seed

            st.session_state["client"] = InferenceClient(
                token=st.secrets.get("hf_token"), model=model
            )
            st.session_state["generation_config"] = generation_config

            LOGGER.warning(f"FORM {model=}")
            LOGGER.warning(f"FORM {generation_config=}")


tab1, tab2, tab3 = st.tabs(["Evaluation", "Training Dataset", "Playground"])

with tab1:
    with st.form("prompt_form"):
        prompt_template = st.text_area(
            "Prompt Template", STARTER_PROMPT, height=PROMPT_TEXT_HEIGHT
        )

        col1, col2 = st.columns(2)

        with col1:
            search_row = st.selectbox(
                "Search label at which row", list(SEARCH_ROW_DICT)
            )

        with col2:
            submitted = st.form_submit_button("Evaluate")

        if submitted:
            if not prompt_template:
                st.error("Prompt template must be specified.")
                st.stop()

            _, formats, *_ = zip(*string.Formatter().parse(prompt_template))
            is_valid_prompt_template = set(formats) == {"text"} or set(formats) == {
                "text",
                None,
            }
            if not is_valid_prompt_template:
                st.error("Prompt template must contain a single {text} field.")
                st.stop()

            inference_progress = st.progress(0, "Executing inference")

            try:
                evaluation = run_evaluation(
                    prompt_template,
                    st.session_state.test_dataset,
                    search_row,
                    st.session_state["generation_config"],
                    inference_progress,
                )
            except HfHubHTTPError as e:
                st.error(e)
                st.stop()

            st.metric("Accuracy", f"{100 * evaluation['accuracy']:.0f}%")

            st.markdown("## Confusion Matrix")
            st.pyplot(evaluation["confusion_matrix_display"])

            st.markdown("## Hits and Misses")
            st.dataframe(evaluation["hit_miss"])

            if evaluation["accuracy"] == 1:
                st.balloons()

with tab2:
    st.dataframe(st.session_state.train_dataset)

with tab3:
    prompt = st.text_area("Prompt", height=PROMPT_TEXT_HEIGHT)

    submitted = st.button("Complete")

    if submitted:
        if not prompt:
            st.error("Prompt must be specified.")
            st.stop()

        with st.spinner("Generating..."):
            output = complete(prompt, st.session_state["generation_config"])

        st.write(output)
