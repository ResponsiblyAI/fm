"""Prompter."""

import logging
import os
import string

import numpy as np
import openai
import pandas as pd
import streamlit as st
from datasets import load_dataset
from datasets.tasks.text_classification import ClassLabel
from huggingface_hub import InferenceClient, dataset_info, model_info
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedShuffleSplit
from spacy.lang.en import English
from tenacity import retry, stop_after_attempt, wait_random_exponential

LOGGER = logging.getLogger(__name__)

TITLE = "Prompter"

OPENAI_API_KEY = st.secrets.get("openai_api_key", None)
HF_TOKEN = st.secrets.get("hf_token", None)

HF_MODEL = os.environ.get("FM_MODEL", "")

HF_DATASET = os.environ.get("FM_HF_DATASET", "")

DATASET_SPLIT_SEED_DEFAULT = 42
TRAIN_SIZE = int(os.environ.get("FM_TRAIN_SIZE", None))
TEST_SIZE = int(os.environ.get("FM_TEST_SIZE", None))
SPLITS = ["train", "test"]
STRATIFY = True

RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60
RETRY_MAX_ATTEMPTS = 6

PROMPT_TEXT_HEIGHT = 300

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
        "END": 1024,
        "DEFAULT": 16,
        "STEP": 16,
        "SAMPLING": False,
    },
    "do_sample": {
        "NAME": "Sampling",
        "DEFAULT": False,
    },
    "stop_sequences": {
        "NAME": "Stop Sequences",
        "DEFAULT": os.environ.get("FM_STOP_SEQUENCES", "").split(),
        "SAMPLING": False,
    },
}

GENERATION_CONFIG_DEFAULTS = {
    key: value["DEFAULT"] for key, value in GENERATION_CONFIG_PARAMS.items()
}

st.set_page_config(page_title=TITLE, initial_sidebar_state="collapsed")


def get_processing_tokenizer():
    return English().tokenizer


PROCESSING_TOKENIZER = get_processing_tokenizer()


@st.cache_resource
def build_api_call_function(model, hf_token=None, openai_api_key=None):
    if model.startswith("openai"):
        openai.api_key = openai_api_key
        _, model = model.split("/")
        openai_models = {model_obj["id"] for model_obj in openai.Model.list()["data"]}
        assert model in openai_models

        @retry(
            wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        )
        def api_call_function(prompt, generation_config):
            if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=generation_config["temperature"],
                    top_p=generation_config["top_p"],
                    max_tokens=generation_config["max_new_tokens"],
                )
                assert response["choices"][0]["message"]["role"] == "assistant"
                output = response["choices"][0]["message"]["content"]

            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    temperature=generation_config["temperature"],
                    top_p=generation_config["top_p"],
                    max_tokens=generation_config["max_new_tokens"],
                    stop=generation_config["stop_sequences"],
                )
                output = response.choices[0].text

            try:
                length = response.total_tokens
            except AttributeError:
                length = None

            return output, length

    else:
        hf_client = InferenceClient(token=hf_token, model=model)

        @retry(
            wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
        )
        def api_call_function(prompt, generation_config):
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

            response = hf_client.text_generation(
                prompt, stream=False, details=True, **generation_config
            )
            LOGGER.warning(response)

            length = len(response.details.prefill) + len(response.details.tokens)

            output = response.generated_text

            # response = st.session_state.client.post(json={"inputs": prompt})
            # output = response.json()[0]["generated_text"]
            # output = st.session_state.client.conversational(prompt, model=model)
            # output = output if "https" in st.session_state.client.model else output[len(prompt) :]

            # Remove stop sequences from the output
            # Inspired by
            # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py
            # https://huggingface.co/spaces/tiiuae/falcon-chat/blob/main/app.py
            if (
                "stop_sequences" in generation_config
                and generation_config["stop_sequences"] is not None
            ):
                for stop_sequence in generation_config["stop_sequences"]:
                    output = output.rsplit(stop_sequence, maxsplit=1)[0]

            return output, length

    return api_call_function


def strip_newline_space(text):
    return text.strip("\n").strip()


def normalize(text):
    return strip_newline_space(text).lower().capitalize()


@st.cache_data
def prepare_datasets(
    dataset_name,
    take_split="train",
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    stratify=STRATIFY,
    dataset_split_seed=None,
):
    try:
        ds = load_dataset(dataset_name)
    except FileNotFoundError as e:
        try:
            assert "/" in dataset_name
            dataset_name, subset_name = dataset_name.rsplit("/", 1)
            ds = load_dataset(dataset_name, subset_name)
        except (FileNotFoundError, AssertionError):
            st.error(f"Dataset `{dataset_name}` not found.")
            st.stop()

    label_columns = [
        (name, info)
        for name, info in ds["train"].features.items()
        if isinstance(info, ClassLabel)
    ]
    assert len(label_columns) == 1
    label_column, label_column_info = label_columns[0]
    labels = [normalize(label) for label in label_column_info.names]
    label_dict = dict(enumerate(labels))

    if any(len(PROCESSING_TOKENIZER(label)) > 1 for label in labels):
        st.error(
            "Labels are not single words. "
            "Matching labels won't not work as expected."
        )

    original_input_columns = [
        name
        for name, info in ds["train"].features.items()
        if not isinstance(info, ClassLabel) and info.dtype == "string"
    ]

    input_columns = []
    for input_column in original_input_columns:
        lowered_input_column = input_column.lower()
        if input_column != lowered_input_column:
            ds = ds.rename_column(input_column, lowered_input_column)
        input_columns.append(lowered_input_column)

    df = ds[take_split].to_pandas()
    for input_column in input_columns:
        df[input_column] = df[input_column].apply(strip_newline_space)
    df[label_column] = df[label_column].replace(label_dict)

    df = df[[label_column] + input_columns]

    if train_size is not None and test_size is not None:
        undersample = RandomUnderSampler(
            sampling_strategy="not minority", random_state=dataset_split_seed
        )
        df, df[label_column] = undersample.fit_resample(df, df[label_column])
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            test_size=test_size,
            random_state=dataset_split_seed,
        )
        train_index, test_index = next(iter(sss.split(df, df[label_column])))

        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        dfs = {"train": train_df, "test": test_df}

    else:
        dfs = {take_split: df}

    return dataset_name, dfs, input_columns, label_column, labels


def complete(api_call_function, prompt, generation_config=None):
    if generation_config is None:
        generation_config = {}

    LOGGER.warning(f"API Call\n\n``{prompt}``\n\n{generation_config=}")

    output, length = api_call_function(prompt, generation_config)

    return output, length


def infer(api_call_function, prompt_template, inputs, generation_config=None):
    prompt = prompt_template.format(**inputs)
    output, length = complete(api_call_function, prompt, generation_config)
    return output, prompt, length


def infer_multi(
    api_call_function, prompt_template, inputs_df, generation_config=None, progress=None
):
    props = (i / len(inputs_df) for i in range(1, len(inputs_df) + 1))

    def infer_with_progress(inputs):
        output, prompt, length = infer(
            api_call_function, prompt_template, inputs.to_dict(), generation_config
        )
        if progress is not None:
            progress.progress(next(props))
        return output, prompt, length

    return zip(*inputs_df.apply(infer_with_progress, axis=1))


def preprocess_output_line(text):
    return [
        normalize(token_str)
        for token in PROCESSING_TOKENIZER(text)
        if (token_str := str(token))
    ]


# Inspired by OpenAI depcriated classification endpoint API
# They take the label from the first line of the output
# https://github.com/openai/openai-cookbook/blob/main/transition_guides_for_deprecated_API_endpoints/classification_functionality_example.py
# https://help.openai.com/en/articles/6272941-classifications-transition-guide#h_e63b71a5c8
# Here we take the label from either the *first* or *last* (for CoT) line of the output
# This is not very robust, but it's a start that doesn't requires asking for a structured output such as JSON
# HELM has more robust processing options, we are not using them, but these are the references:
# https://github.com/stanford-crfm/helm/blob/04a75826ce75835f6d22a7d41ae1487104797964/src/helm/benchmark/metrics/classification_metrics.py
# https://github.com/stanford-crfm/helm/blob/04a75826ce75835f6d22a7d41ae1487104797964/src/helm/benchmark/metrics/basic_metrics.py
def canonize_label(output, annotation_labels, search_row):
    assert search_row in SEARCH_ROW_DICT.keys()

    search_row_index = SEARCH_ROW_DICT[search_row]

    annotation_labels_set = set(annotation_labels)

    output_lines = strip_newline_space(output).split("\n")
    output_search_words = preprocess_output_line(output_lines[search_row_index])

    label_matches = set(output_search_words) & annotation_labels_set

    if len(label_matches) == 1:
        return next(iter(label_matches))
    else:
        return UNKNOWN_LABEL


def measure(dataset, outputs, labels, label_column, input_columns, search_row):
    inferences = [canonize_label(output, labels, search_row) for output in outputs]

    LOGGER.warning(f"{inferences=}")
    LOGGER.warning(f"{labels=}")
    inference_labels = labels + [UNKNOWN_LABEL]

    evaluation_df = pd.DataFrame(
        {
            "hit/miss": np.where(dataset[label_column] == inferences, "hit", "miss"),
            "annotation": dataset[label_column],
            "inference": inferences,
            "output": outputs,
        }
        | dataset[input_columns].to_dict("list")
    )

    acc = accuracy_score(evaluation_df["annotation"], evaluation_df["inference"])
    bacc = balanced_accuracy_score(
        evaluation_df["annotation"], evaluation_df["inference"]
    )
    mcc = matthews_corrcoef(evaluation_df["annotation"], evaluation_df["inference"])
    cm = confusion_matrix(
        evaluation_df["annotation"], evaluation_df["inference"], labels=inference_labels
    )

    cm_display = ConfusionMatrixDisplay(cm, display_labels=inference_labels)
    cm_display.plot()
    cm_display.ax_.set_xlabel("Inference Labels")
    cm_display.ax_.set_ylabel("Annotation Labels")
    cm_display.figure_.autofmt_xdate(rotation=45)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "confusion_matrix": cm,
        "confusion_matrix_display": cm_display.figure_,
        "hit_miss": evaluation_df,
        "annotation_labels": labels,
        "inference_labels": inference_labels,
    }

    return metrics


def run_evaluation(
    api_call_function,
    prompt_template,
    dataset,
    labels,
    label_column,
    input_columns,
    search_row,
    generation_config=None,
    progress=None,
):
    inputs_df = dataset[input_columns]
    outputs, prompts, lengths = infer_multi(
        api_call_function,
        prompt_template,
        inputs_df,
        generation_config,
        progress,
    )

    metrics = measure(dataset, outputs, labels, label_column, input_columns, search_row)

    metrics["hit_miss"]["prompt"] = prompts
    metrics["hit_miss"]["length"] = lengths

    return metrics


def combine_labels(labels):
    return "|".join(f"``{label}``" for label in labels)


def main():
    if "dataset_split_seed" not in st.session_state:
        st.session_state["dataset_split_seed"] = DATASET_SPLIT_SEED_DEFAULT

    if "train_size" not in st.session_state:
        st.session_state["train_size"] = TRAIN_SIZE

    if "test_size" not in st.session_state:
        st.session_state["test_size"] = TEST_SIZE

    if "api_call_function" not in st.session_state:
        st.session_state["api_call_function"] = build_api_call_function(
            model=HF_MODEL, hf_token=HF_TOKEN, openai_api_key=OPENAI_API_KEY
        )

    if "train_dataset" not in st.session_state:
        (
            st.session_state["dataset_name"],
            splits_df,
            st.session_state["input_columns"],
            st.session_state["label_column"],
            st.session_state["labels"],
        ) = prepare_datasets(
            HF_DATASET,
            train_size=st.session_state.train_size,
            test_size=st.session_state.test_size,
            dataset_split_seed=st.session_state.dataset_split_seed,
        )

        for split in splits_df:
            st.session_state[f"{split}_dataset"] = splits_df[split]

    if "generation_config" not in st.session_state:
        st.session_state["generation_config"] = GENERATION_CONFIG_DEFAULTS

    st.title(TITLE)

    with st.sidebar:
        with st.form("model_form"):
            model = st.text_input("Model", HF_MODEL).strip()

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

            do_sample = st.checkbox(
                GENERATION_CONFIG_PARAMS["do_sample"]["NAME"],
                value=GENERATION_CONFIG_PARAMS["do_sample"]["DEFAULT"],
            )

            stop_sequences = st.text_area(
                GENERATION_CONFIG_PARAMS["stop_sequences"]["NAME"],
                value="\n".join(GENERATION_CONFIG_PARAMS["stop_sequences"]["DEFAULT"]),
            )

            stop_sequences = [
                clean_stop.encode().decode("unicode_escape")  # interpret \n as newline
                for stop in stop_sequences.split("\n")
                if (clean_stop := stop.strip())
            ]
            if not stop_sequences:
                stop_sequences = None

            decoding_seed = st.text_input("Decoding Seed").strip()

            st.divider()

            dataset = st.text_input("Dataset", HF_DATASET).strip()

            train_size = st.number_input("Train Size", value=TRAIN_SIZE, min_value=10)
            test_size = st.number_input("Test Size", value=TEST_SIZE, min_value=10)

            stratify = st.checkbox("Stratify", STRATIFY)

            dataset_split_seed = st.text_input(
                "Dataset Split Seed", st.session_state["dataset_split_seed"]
            ).strip()

            st.divider()

            submitted = st.form_submit_button("Set")

            if submitted:
                if not dataset:
                    st.error("Dataset must be specified.")
                    st.stop()

                if not model:
                    st.error("Model must be specified.")
                    st.stop()

                if not decoding_seed:
                    decoding_seed = None
                elif seed.isnumeric():
                    decoding_seed = int(seed)
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
                    and not do_sample
                ):
                    sampling_slider_default_values_info = " | ".join(
                        f"{name}={GENERATION_CONFIG_DEFAULTS[name]}"
                        for name in generation_confing_slider_sampling
                    )
                    st.error(
                        f"Sampling must be enabled to use non default values for generation parameters: {sampling_slider_default_values_info}"
                    )
                    st.stop()

                if decoding_seed is not None and not do_sample:
                    st.error(
                        "Sampling must be enabled to use a decoding seed. Otherwise, the seed field should be empty."
                    )
                    st.stop()

                if not dataset_split_seed:
                    dataset_split_seed = None
                elif dataset_split_seed.isnumeric():
                    dataset_split_seed = int(dataset_split_seed)
                else:
                    st.error("Dataset split seed must be numeric or empty.")
                    st.stop()

                generation_config = generation_config_sliders | dict(
                    do_sample=do_sample,
                    stop_sequences=stop_sequences,
                    seed=decoding_seed,
                )

                st.session_state["dataset_split_seed"] = dataset_split_seed
                st.session_state["train_size"] = train_size
                st.session_state["test_size"] = test_size

                st.session_state["api_call_function"] = build_api_call_function(
                    model=model, hf_token=HF_TOKEN, openai_api_key=OPENAI_API_KEY
                )

                st.session_state["generation_config"] = generation_config

                (
                    st.session_state["dataset_name"],
                    splits_df,
                    st.session_state["input_columns"],
                    st.session_state["label_column"],
                    st.session_state["labels"],
                ) = prepare_datasets(
                    dataset,
                    train_size=st.session_state.train_size,
                    test_size=st.session_state.test_size,
                    stratify=stratify,
                    dataset_split_seed=st.session_state.dataset_split_seed,
                )

                for split in splits_df:
                    st.session_state[f"{split}_dataset"] = splits_df[split]

                LOGGER.warning(f"FORM {dataset=}")
                LOGGER.warning(f"FORM {model=}")
                LOGGER.warning(f"FORM {generation_config=}")

        with st.expander("Info"):
            st.caption("Dataset")
            st.write(dataset_info(st.session_state.dataset_name).cardData)
            try:
                st.caption("Model")
                st.write(model_info(model).cardData)
            except RepositoryNotFoundError:
                pass

            # st.write(f"Model max length: {AutoTokenizer.from_pretrained(model).model_max_length}")

    tab1, tab2, tab3 = st.tabs(["Evaluation", "Training Dataset", "Playground"])

    with tab1:
        with st.form("prompt_form"):
            prompt_template = st.text_area("Prompt Template", height=PROMPT_TEXT_HEIGHT)

            st.write(f"Labels: {combine_labels(st.session_state.labels)}")
            st.write(f"Inputs: {combine_labels(st.session_state.input_columns)}")

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
                is_valid_prompt_template = set(formats).issubset(
                    {None} | set(st.session_state.input_columns)
                )

                if not is_valid_prompt_template:
                    st.error(f"The prompt template contains unrecognized fields.")
                    st.stop()

                inference_progress = st.progress(0, "Executing inference")

                try:
                    evaluation = run_evaluation(
                        st.session_state.api_call_function,
                        prompt_template,
                        st.session_state.test_dataset,
                        st.session_state.labels,
                        st.session_state.label_column,
                        st.session_state.input_columns,
                        search_row,
                        st.session_state.generation_config,
                        inference_progress,
                    )
                except HfHubHTTPError as e:
                    st.error(e)
                    st.stop()

                num_metric_cols = 1 if stratify else 3
                cols = st.columns(num_metric_cols)
                with cols[0]:
                    st.metric("Accuracy", f"{100 * evaluation['accuracy']:.0f}%")
                if not stratify:
                    with cols[1]:
                        st.metric(
                            "Balanced Accuracy",
                            f"{100 * evaluation['balanced_accuracy']:.0f}%",
                        )
                    with cols[2]:
                        st.metric("MCC", f"{evaluation['mcc']:.2f}")

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
                try:
                    output, _ = complete(
                        st.session_state.api_call_function,
                        prompt,
                        st.session_state.generation_config,
                    )
                except HfHubHTTPError as e:
                    st.error(e)
                    st.stop()

            st.text(output)


if __name__ == "__main__":
    main()
