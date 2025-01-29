from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

CONFIG_FILE = Path(__file__).parent / "model_configs.yaml"


class StringInFile(BaseModel):
    path: str
    _string: str = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_and_extract_string(self):
        full_path = Path(CONFIG_FILE).parent / self.path
        if full_path.exists():
            with open(full_path, "r") as f:
                self._string = f.read()
        else:
            raise ValueError("Invalid path")
        return self

    @property
    def string(self):
        return self._string


class ModelConfig(BaseModel):
    model_id: str
    name: Union[str, StringInFile] = Field(default="")
    system_prompt: Optional[str] = None
    user_template: Optional[str] = None

    @field_validator("name", mode="before")
    def validate_name(cls, v):
        if v is None:
            return cls.model_id.split("/")[-1]
        return v

    @field_validator("system_prompt", mode="before")
    def validate_system_prompt(cls, v):
        if v is None:
            return v


if __name__ == "__main__":
    s = StringInFile(path="prime.txt")
    breakpoint()
