import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import BaseModel, Field

from ..models import ModelConfig

MessagesType = List[Dict[str, str]]


class PreprocessConfig(BaseModel, extra="allow"):
    difficulty: str


class TaskConfig(BaseModel, extra="forbid"):
    handler: str
    dataset_path: str
    dataset_source: Optional[str] = None
    dataset_split: str
    dataset_kwargs: Dict[str, Any] = Field(default_factory=dict)
    question_key: str
    # Optional answer key for datasets with a single correct answer
    answer_key: Optional[str] = None
    templating_parameters: Dict[str, str] = Field(default_factory=dict)
    # Optional, unused for now
    fewshot_config: List[Dict[str, Any]] = Field(default_factory=list)
    num_fewshot: int = 0

    preprocess_config: Optional[PreprocessConfig] = None

    @classmethod
    def from_yaml(cls, yaml_file_path) -> "TaskConfig":
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class TaskHandler(ABC):

    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config

    @classmethod
    def from_config_path(cls, config_path: str) -> "TaskHandler":
        """Instantiates a TaskHandler from a config file path

        Args:
            config_path (str): Path to the config file

        Returns:
            TaskHandler: The instantiated TaskHandler
        """
        task_config = TaskConfig.from_yaml(config_path)
        return cls(task_config)

    @property
    def question_key(self):
        return self.task_config.question_key

    @abstractmethod
    def check_correctness(self, problem: str, generation: str) -> bool:
        """Checks the correctness of a generation for a given problem

        Args:
            problem (str): The problem to check
            generation (str): The generation to check

        Returns:
            bool: Whether the generation is correct
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def update_results(self, problem, response):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def make_conversations(self, data, model_config: ModelConfig):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_existing_results(self, result_file):
        if not os.path.exists(result_file):
            return {}
        with open(result_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records

    def load_dataset(self, source=None, split=None) -> HFDataset:
        dataset = load_dataset(
            path=self.task_config.dataset_path,
            name=source if source else self.task_config.dataset_source,
            split=split if split else self.task_config.dataset_split,
            **self.task_config.dataset_kwargs
        )
        return dataset

    @staticmethod
    def format_into_conversation(
        contents: List[str], system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Formats a list of message contents and an optional system prompt in the OpenAI conversational format

        Assumes that `contents` has a list of alternating user and assistant messages (i.e u/a/u/a....)
        """
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})

        for i, content in enumerate(contents):
            if i % 2 == 0:
                conversation.append({"role": "user", "content": content})
            else:
                conversation.append({"role": "assistant", "content": content})
        return conversation

    @abstractmethod
    def load_and_filter_dataset(
        self, start, end, split=None, source=None, filter_difficulty=None, args=None
    ):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def process_remaining_data(self, train_data, results):
        raise NotImplementedError("Subclasses should implement this method.")
