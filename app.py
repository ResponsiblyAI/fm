"""Prompter."""

import asyncio
import logging
import os
import sys
import importlib
import string

import aiohttp
import cohere
import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_dataset
from datasets.features import ClassLabel
from huggingface_hub import AsyncInferenceClient, dataset_info, model_info
from huggingface_hub.utils import (
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
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
from transformers import pipeline

HOW_OPENAI_INITIATED = None

LOGGER = logging.getLogger(__name__)

TITLE = "Prompter"

OPENAI_API_KEY = st.secrets.get("openai_api_key", None)
TOGETHER_API_KEY = st.secrets.get("together_api_key", None)
HF_TOKEN = st.secrets.get("hf_token", None)
COHERE_API_KEY = st.secrets.get("cohere_api_key", None)
AZURE_OPENAI_KEY = st.secrets.get("azure_openai_key", None)
AZURE_OPENAI_ENDPOINT = st.secrets.get("azure_openai_endpoint", None)
AZURE_DEPLOYMENT_NAME = st.secrets.get("azure_deployment_name", None)

HF_MODEL = os.environ.get("FM_MODEL", "")

HF_DATASET = os.environ.get("FM_HF_DATASET", "")

DATASET_SPLIT_SEED = os.environ.get("FM_DATASET_SPLIT_SEED", "")
TRAIN_SIZE = 15
TEST_SIZE = 25
BALANCING = True

RETRY_MIN_WAIT = 1
RETRY_MAX_WAIT = 60
RETRY_MAX_ATTEMPTS = 6

PROMPT_TEXT_HEIGHT = 300

UNKNOWN_LABEL = "Unknown"

SEARCH_ROW_DICT = {"First": 0, "Last": -1}

KNOWN_PROVIDERS = ("openai", "azure", "together", "cohere", "@")

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
    "is_chat": {
        "NAME": "Is Chat",
        "DEFAULT": False,
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


class OpenAIAlreadyInitiatedError(Exception):
    """OpenAIAlreadyInitiatedError."""

    pass


def prepare_huggingface_generation_config(generation_config):
    generation_config = generation_config.copy()

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

    # TODO: refactor this part
    if generation_config["is_chat"]:
        generation_config["max_tokens"] = generation_config.pop("max_new_tokens")

        generation_config["stop"] = generation_config.pop("stop_sequences")
        del generation_config["do_sample"]
        del generation_config["top_k"]

    is_chat = generation_config.pop("is_chat")

    return generation_config, is_chat


def escape_markdown(text):
    escape_dict = {
        "*": r"\*",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "[": r"\[",
        "]": r"\]",
        "(": r"\(",
        ")": r"\)",
        "+": r"\+",
        "-": r"\-",
        ".": r"\.",
        "!": r"\!",
        "`": r"\`",
        ">": r"\>",
        "|": r"\|",
        "#": r"\#",
    }
    return "".join([escape_dict.get(c, c) for c in text])


def reload_module(name):
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def build_api_call_function(model):
    global HOW_OPENAI_INITIATED

    if any(
        model.startswith(known_providers)
        for known_providers in KNOWN_PROVIDERS
    ):
        provider, model = model.split("/", maxsplit=1)

        if provider == "openai":
            from openai import AsyncOpenAI

            aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

        elif provider == "azure":
            from openai import AsyncAzureOpenAI

            aclient = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
                api_version="2025-04-01-preview",
                # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )

        elif provider == "together":
            from openai import AsyncOpenAI

            aclient = AsyncOpenAI(
                api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1"
            )

        if provider in ("openai", "azure"):

            async def list_models():
                return [model async for model in aclient.models.list()]

            openai_models = {model_obj.id for model_obj in asyncio.run(list_models())}
            assert model in openai_models

        @retry(
            wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            reraise=True,
        )
        async def api_call_function(prompt, generation_config):
            temperature = (
                generation_config["temperature"]
                if generation_config["do_sample"]
                else 0
            )
            top_p = generation_config["top_p"] if generation_config["do_sample"] else 1
            max_tokens = generation_config["max_new_tokens"]

            if (
                model.startswith("gpt") and "instruct" not in model
            ) or provider == "together":
                response = await aclient.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                assert response.choices[0].message.role == "assistant"
                output = response.choices[0].message.content

            else:
                response = await aclient.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                output = response.choices[0].text

            try:
                length = response.usage.total_tokens
            except AttributeError:
                length = None

            return output, length

    elif model.startswith("cohere"):
        _, model = model.split("/")

        @retry(
            wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            reraise=True,
        )
        async def api_call_function(prompt, generation_config):
            async with cohere.AsyncClient(COHERE_API_KEY) as co:
                response = await co.generate(
                    model=model,
                    prompt=prompt,
                    temperature=generation_config["temperature"]
                    if generation_config["do_sample"]
                    else 0,
                    p=generation_config["top_p"]
                    if generation_config["do_sample"]
                    else 1,
                    k=generation_config["top_k"]
                    if generation_config["do_sample"]
                    else 0,
                    max_tokens=generation_config["max_new_tokens"],
                    end_sequences=generation_config["stop_sequences"],
                )

            output = response.generations[0].text
            length = None

            return output, length

    elif model.startswith("@"):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is not available. Please install it to use local models (models starting with '@').")
        
        model = model[1:]
        pipe = pipeline(
            "text-generation", model=model, trust_remote_code=True, device_map="auto"
        )

        async def api_call_function(prompt, generation_config):
            generation_config, _ = prepare_huggingface_generation_config(
                generation_config
            )

            # TODO: include chat
            output = pipe(prompt, return_text=True, **generation_config)[0][
                "generated_text"
            ]
            output = output[len(prompt) :]

            length = None

            return output, length

    else:

        @retry(
            wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
            stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
            reraise=True,
        )
        async def api_call_function(prompt, generation_config):
            hf_client = AsyncInferenceClient(token=HF_TOKEN, model=model)

            generation_config, is_chat = prepare_huggingface_generation_config(
                generation_config
            )

            if is_chat:
                messages = [{"role": "user", "content": prompt}]
                response = await hf_client.chat_completion(
                    messages, stream=False, **generation_config
                )
                output = response.choices[0].message.content
                length = None

            else:
                response = await hf_client.text_generation(
                    prompt, stream=False, details=True, **generation_config
                )

                length = (
                    len(response.details.prefill) + len(response.details.tokens)
                    if response.details is not None
                    else None
                )

                output = response.generated_text

            # TODO: refactor to support stop of chats
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


def prepare_datasets(
    dataset_name,
    take_split="train",
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    balancing=BALANCING,
    dataset_split_seed=None,
):
    try:
        ds = load_dataset(dataset_name, trust_remote_code=True)
    except FileNotFoundError as e:
        try:
            assert "/" in dataset_name
            dataset_name, subset_name = dataset_name.rsplit("/", 1)
            ds = load_dataset(dataset_name, subset_name, trust_remote_code=True)
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
        # Try to apply undersampling, but fallback if there are compatibility issues
        try:
            undersample = RandomUnderSampler(
                sampling_strategy="not minority", random_state=dataset_split_seed
            )
            df, df[label_column] = undersample.fit_resample(df, df[label_column])
        except Exception as e:
            # Log the error and continue without undersampling
            LOGGER.warning(f"Undersampling failed: {e}. Proceeding without undersampling.")
        
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


async def complete(api_call_function, prompt, generation_config=None):
    if generation_config is None:
        generation_config = GENERATION_CONFIG_DEFAULTS.copy()

    LOGGER.info(f"API Call\n\n``{prompt}``\n\n{generation_config=}")

    output, length = await api_call_function(prompt, generation_config)

    return output, length


async def infer(api_call_function, prompt_template, inputs, generation_config=None):
    prompt = prompt_template.format(**inputs)
    output, length = await complete(api_call_function, prompt, generation_config)
    return output, prompt, length


async def infer_multi(
    api_call_function, prompt_template, inputs_df, generation_config=None
):
    results = await asyncio.gather(
        *(
            infer(
                api_call_function, prompt_template, inputs.to_dict(), generation_config
            )
            for _, inputs in inputs_df.iterrows()
        )
    )

    return zip(*results)


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

    LOGGER.info(f"{inferences=}")
    LOGGER.info(f"{labels=}")
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

    unknown_proportion = (evaluation_df["inference"] == UNKNOWN_LABEL).mean()

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
        "unknown_proportion": unknown_proportion,
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
):
    inputs_df = dataset[input_columns]
    outputs, prompts, lengths = asyncio.run(
        infer_multi(
            api_call_function,
            prompt_template,
            inputs_df,
            generation_config,
        )
    )

    metrics = measure(dataset, outputs, labels, label_column, input_columns, search_row)

    metrics["hit_miss"]["prompt"] = prompts
    metrics["hit_miss"]["length"] = lengths

    return metrics


def combine_labels(labels):
    return "|".join(f"``{label}``" for label in labels)


def main():
    try:
        if "dataset_split_seed" not in st.session_state:
            st.session_state["dataset_split_seed"] = (
                int(DATASET_SPLIT_SEED) if DATASET_SPLIT_SEED else None
            )

        if "train_size" not in st.session_state:
            st.session_state["train_size"] = TRAIN_SIZE

        if "test_size" not in st.session_state:
            st.session_state["test_size"] = TEST_SIZE

        if "api_call_function" not in st.session_state:
            st.session_state["api_call_function"] = build_api_call_function(
                model=HF_MODEL,
            )

        if "generation_config" not in st.session_state:
            st.session_state["generation_config"] = GENERATION_CONFIG_DEFAULTS

        # Try to initialize dataset, but don't fail the entire app if it doesn't work
        if "train_dataset" not in st.session_state and HF_DATASET:
            try:
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
            except Exception as dataset_error:
                st.error(f"Failed to load dataset '{HF_DATASET}': {dataset_error}")
                st.info("You can manually set the dataset in the sidebar form.")

    except Exception as e:
        st.error(f"Initialization error: {e}")
        st.info("Please configure the model and dataset manually in the sidebar.")

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

            st.markdown("---")

            dataset = st.text_input("Dataset", HF_DATASET).strip()

            train_size = st.number_input("Train Size", value=TRAIN_SIZE, min_value=10)
            test_size = st.number_input("Test Size", value=TEST_SIZE, min_value=10)

            balancing = st.checkbox("Balancing", BALANCING)

            dataset_split_seed = st.text_input(
                "Dataset Split Seed", DATASET_SPLIT_SEED
            ).strip()

            st.markdown("---")

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
                elif decoding_seed.isnumeric():
                    decoding_seed = int(decoding_seed)
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
                    is_chat=False,  # Will be updated later based on model info
                )

                st.session_state["dataset_split_seed"] = dataset_split_seed
                st.session_state["train_size"] = train_size
                st.session_state["test_size"] = test_size

                try:
                    st.session_state["api_call_function"] = build_api_call_function(
                        model=model,
                    )
                except OpenAIAlreadyInitiatedError as e:
                    st.error(e)
                    st.stop()

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
                    balancing=balancing,
                    dataset_split_seed=st.session_state.dataset_split_seed,
                )

                for split in splits_df:
                    st.session_state[f"{split}_dataset"] = splits_df[split]

                LOGGER.info(f"FORM {dataset=}")
                LOGGER.info(f"FORM {model=}")
                LOGGER.info(f"FORM {generation_config=}")

        with st.expander("Info"):
            data_card = None  # Initialize variable
            try:
                # Check if dataset_name exists in session state before accessing it
                if hasattr(st.session_state, 'dataset_name'):
                    data_card = dataset_info(st.session_state.dataset_name, token=HF_TOKEN).cardData
            except (HFValidationError, RepositoryNotFoundError):
                pass
            
            # Display dataset info if available
            if data_card:
                st.caption("Dataset")
                try:
                    # Try to convert to dict if it's not already
                    if hasattr(data_card, '__dict__'):
                        data_dict = data_card.__dict__
                    else:
                        data_dict = dict(data_card) if data_card else {}
                    st.json(data_dict)
                except (TypeError, ValueError) as e:
                    # Fallback to text representation if JSON serialization fails
                    st.text(str(data_card))
            
            # Only try to get model info from HuggingFace for non-provider models
            model_card = None  # Initialize variable
            if not any(model.startswith(known_provider) for known_provider in KNOWN_PROVIDERS):
                try:
                    model_info_respose = model_info(model)
                    model_card = model_info_respose.cardData
                    st.session_state["generation_config"]["is_chat"] = (
                        "conversational" in model_info_respose.tags
                    )
                except (HFValidationError, RepositoryNotFoundError):
                    pass
                
                # Display model info if available
                if model_card:
                    st.caption("Model")
                    try:
                        # Try to convert to dict if it's not already
                        if hasattr(model_card, '__dict__'):
                            model_dict = model_card.__dict__
                        else:
                            model_dict = dict(model_card) if model_card else {}
                        st.json(model_dict)
                    except (TypeError, ValueError) as e:
                        # Fallback to text representation if JSON serialization fails
                        st.text(str(model_card))
            else:
                # For provider models, we can't get info from HuggingFace
                # Set reasonable defaults for chat detection
                if model.startswith("openai/gpt") and "instruct" not in model:
                    st.session_state["generation_config"]["is_chat"] = True
                elif model.startswith("azure/gpt") and "instruct" not in model:
                    st.session_state["generation_config"]["is_chat"] = True
                elif model.startswith("together/") and any(chat_indicator in model for chat_indicator in ["chat", "instruct"]):
                    st.session_state["generation_config"]["is_chat"] = True
                else:
                    st.session_state["generation_config"]["is_chat"] = False

    tab1, tab2, tab3 = st.tabs(["Evaluation", "Examples", "Playground"])

    with tab1:
        # Check if required session state variables exist before proceeding
        if not all(hasattr(st.session_state, attr) for attr in ['input_columns', 'labels']):
            st.warning("Dataset not loaded. Please configure the dataset in the sidebar and click 'Set'.")
        else:
            with st.form("prompt_form"):
                prompt_template = st.text_area("Prompt Template", height=PROMPT_TEXT_HEIGHT)

                is_multi_placeholder = len(st.session_state.input_columns) > 1

                st.write(
                    f"To determine the inferred label of an input, the model should output one of the following words:"
                    f" {combine_labels(st.session_state.labels)}"
                )
                st.write(
                    f"The input placeholder{'s' if is_multi_placeholder else ''} available for the prompt template {'are' if is_multi_placeholder else 'is'}:"
                    f" {combine_labels(f'{{{col}}}' for col in st.session_state.input_columns)}"
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
                    is_valid_prompt_template = set(formats).issubset(
                        {None} | set(st.session_state.input_columns)
                    )

                    if not is_valid_prompt_template:
                        st.error(f"The prompt template contains unrecognized fields.")
                        st.stop()

                    with st.spinner("Executing inference..."):
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
                            )
                        except HfHubHTTPError as e:
                            st.error(e)
                            st.stop()

                    st.markdown("### Metrics")
                    num_metric_cols = 2 if balancing else 4
                    cols = st.columns(num_metric_cols)
                    with cols[0]:
                        st.metric("Accuracy", f"{100 * evaluation['accuracy']:.0f}%")
                        st.caption("The percentage of correct inferences.")
                    with cols[1]:
                        st.metric(
                            "Unknown",
                            f"{100 * evaluation['unknown_proportion']:.0f}%",
                        )
                        st.caption(
                            "The percentage of inferences"
                            " that could not be determined based on the model output."
                        )
                    if not balancing:
                        with cols[2]:
                            st.metric(
                                "Balanced Accuracy",
                                f"{100 * evaluation['balanced_accuracy']:.0f}%",
                            )
                        with cols[3]:
                            st.metric("MCC", f"{evaluation['mcc']:.2f}")

                    st.markdown("### Detailed Evaluation")

                    st.caption(
                        "This table showcases all examples (input and output pairs) that were leveraged for the evaluation of the prompt template with the model (for instance, accuracy)."
                        " It comprises the input placeholder values, the unmodified model *output*, the deduced *inference*, and the ground-truth *annotation*."
                    )
                    st.caption(
                        "A 'hit' signifies a correct inference (when *inference* coincides with *annotation*), while a 'miss' denotes an incorrect inference."
                        " If the *inference* cannot be determined based on the model output, it is labeled as 'unknown'."
                    )
                    st.caption(
                        "The *prompt* column features the complete prompt that the model was prompted to complete, i.e., your prompt template filled with the input placeholders you have used."
                    )
                    st.caption(
                        "You are not allowed to include these examples in your prompt template."
                    )

                    st.dataframe(evaluation["hit_miss"])

                    with st.expander("Additional Information", expanded=False):
                        st.markdown("## Confusion Matrix")
                        st.pyplot(evaluation["confusion_matrix_display"])

                    if evaluation["accuracy"] == 1:
                        st.balloons()

    with tab2:
        if not hasattr(st.session_state, 'train_dataset'):
            st.warning("Training dataset not loaded. Please configure the dataset in the sidebar and click 'Set'.")
        else:
            st.caption(
                "You can include the following examples in your prompt template for few-shot prompting."
            )
            st.dataframe(st.session_state.train_dataset)

    with tab3:
        if not all(hasattr(st.session_state, attr) for attr in ['api_call_function', 'generation_config']):
            st.warning("Model not configured. Please set up the model in the sidebar and click 'Set'.")
        else:
            prompt = st.text_area("Prompt", height=PROMPT_TEXT_HEIGHT)

            submitted = st.button("Complete")

            if submitted:
                if not prompt:
                    st.error("Prompt must be specified.")
                    st.stop()

                with st.spinner("Generating..."):
                    try:
                        output, length = asyncio.run(
                            complete(
                                st.session_state.api_call_function,
                                prompt,
                                st.session_state.generation_config,
                            )
                        )
                    except HfHubHTTPError as e:
                        st.error(e)
                        st.stop()
                st.markdown(escape_markdown(output))
                if length is not None:
                    with st.expander("Stats"):
                        st.metric("#Tokens", length)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
