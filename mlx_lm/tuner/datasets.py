import itertools
import json
import types
from pathlib import Path
from typing import Any, Dict, List, Union

from transformers import PreTrainedTokenizer


class ORPODataset:
    def __init__(
        self,
        data: List[Dict[str, Union[str, Dict, List]]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        preference_score_key: str = "preference_score",
        system_key: str = None,
    ):
        self._chosen_data = []
        self._rejected_data = []
        self._scores = []

        for d in data:
            prompt_content = d.get(prompt_key, d.get("question", ""))

            if system_key and system_key in d:
                base_messages = [{"role": "system", "content": d[system_key]}]
                chosen_messages = base_messages + [
                    {"role": "user", "content": prompt_content}
                ]
                rejected_messages = base_messages + [
                    {"role": "user", "content": prompt_content}
                ]

                if isinstance(d[chosen_key], str):
                    chosen_messages.append(
                        {"role": "assistant", "content": d[chosen_key]}
                    )
                elif isinstance(d[chosen_key], dict):
                    if "messages" in d[chosen_key]:
                        chosen_messages.extend(d[chosen_key]["messages"])
                    else:
                        chosen_messages.append(
                            {
                                "role": "assistant",
                                "content": d[chosen_key].get("content", ""),
                            }
                        )
                elif isinstance(d[chosen_key], list):
                    chosen_messages.extend(d[chosen_key])

                if isinstance(d[rejected_key], str):
                    rejected_messages.append(
                        {"role": "assistant", "content": d[rejected_key]}
                    )
                elif isinstance(d[rejected_key], dict):
                    if "messages" in d[rejected_key]:
                        rejected_messages.extend(d[rejected_key]["messages"])
                    else:
                        rejected_messages.append(
                            {
                                "role": "assistant",
                                "content": d[rejected_key].get("content", ""),
                            }
                        )
                elif isinstance(d[rejected_key], list):
                    rejected_messages.extend(d[rejected_key])

                chosen_text = tokenizer.apply_chat_template(chosen_messages)
                rejected_text = tokenizer.apply_chat_template(rejected_messages)

            else:
                chosen_content = self._extract_content(d[chosen_key])
                rejected_content = self._extract_content(d[rejected_key])

                chosen_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": chosen_content},
                    ]
                )
                rejected_text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt_content},
                        {"role": "assistant", "content": rejected_content},
                    ]
                )

            self._chosen_data.append(chosen_text)
            self._rejected_data.append(rejected_text)

            if preference_score_key in d:
                self._scores.append(float(d[preference_score_key]))
            else:
                self._scores.append(1.0)

    def _extract_content(self, data):
        """Helper method to extract content from various data formats."""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            if "messages" in data:
                last_message = data["messages"][-1]
                return last_message.get("content", last_message.get("messages", ""))
            return data.get("content", "")
        elif isinstance(data, list):
            last_message = data[-1]
            if isinstance(last_message, dict):
                if "content" in last_message:
                    return last_message["content"]
                elif "messages" in last_message:
                    return last_message["messages"]
            return last_message if isinstance(last_message, str) else ""
        return ""

    def __len__(self):
        return len(self._chosen_data)

    def __getitem__(self, idx: int):
        return {
            "chosen": self._chosen_data[idx],
            "rejected": self._rejected_data[idx],
            "preference_score": self._scores[idx],
        }


class Dataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = [tokenizer.encode(d[text_key]) for d in data]
        for d in self._data:
            if d[-1] != tokenizer.eos_token_id:
                d.append(tokenizer.eos_token_id)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
        mask_prompt: bool = False,
    ):
        self._data = []
        for d in data:
            messages = d[chat_key]
            tools = d.get("tools", None)
            tokens = tokenizer.apply_chat_template(messages, tools=tools)
            if mask_prompt:
                messages = messages[:-1]
                offset = len(tokenizer.apply_chat_template(messages, tools=tools))
                self._data.append((tokens, offset))
            else:
                self._data.append(tokens)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
        mask_prompt: bool,
    ):
        self._data = []
        for d in data:
            tokens = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": d[prompt_key]},
                    {"role": "assistant", "content": d[completion_key]},
                ],
            )
            if mask_prompt:
                offset = len(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": d[prompt_key]}]
                    )
                )
                self._data.append((tokens, offset))
            else:
                self._data.append(tokens)

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ConcatenatedDataset:
    def __init__(self, data: List[Any]):
        self._data = list(itertools.chain(*data))

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def create_dataset(
    data,
    tokenizer: PreTrainedTokenizer,
    config,
):
    mask_prompt = getattr(config, "mask_prompt", False)
    prompt_feature = getattr(config, "prompt_feature", "prompt")
    text_feature = getattr(config, "text_feature", "text")
    completion_feature = getattr(config, "completion_feature", "completion")
    chat_feature = getattr(config, "chat_feature", "messages")
    training_mode = getattr(config, "training_mode", "normal")
    sample = data[0]
    if training_mode == "normal":
        if prompt_feature in sample and completion_feature in sample:
            return CompletionsDataset(
                data, tokenizer, prompt_feature, completion_feature, mask_prompt
            )
        elif chat_feature in sample:
            return ChatDataset(
                data, tokenizer, chat_key=chat_feature, mask_prompt=mask_prompt
            )
        elif text_feature in sample:
            if mask_prompt:
                raise ValueError("Prompt masking not supported for text dataset.")
            return Dataset(data, tokenizer, text_key=text_feature)
        else:
            raise ValueError(
                "Unsupported data format, check the supported formats here:\n"
                "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
            )
    else:
        if "chosen" in sample and "rejected" in sample:
            return ORPODataset(data, tokenizer)
        else:
            raise ValueError(
                "Unsupported data format, check the supported formats here:\n"
                "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
            )


def load_local_dataset(
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(data, tokenizer, config)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        names = ("train", "valid", "test")

        train, valid, test = [
            (
                create_dataset(dataset[n], tokenizer, config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    def create_hf_dataset(dataset_name, config, split, hf_config):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_config,
        )
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_name = ds["name"]
        print(f"Loading Hugging Face dataset {ds_name}.")
        ds["mask_prompt"] = getattr(args, "mask_prompt", False)
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})
        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(
                ds_name,
                config,
                train_split,
                hf_config,
            )
            valid = create_hf_dataset(
                ds_name,
                config,
                valid_split,
                hf_config,
            )
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(
                ds_name,
                config,
                test_split,
                hf_config,
            )
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    # Otherwise concatenate them
    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(data_path, tokenizer, args)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args.data, tokenizer, args)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
