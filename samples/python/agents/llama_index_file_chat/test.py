import os
import tempfile
from datetime import datetime
from llama_index.core.tools import FunctionTool
from typing import List

import requests

import tiktoken
from llama_index.llms.openai import OpenAI

from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate

from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock

from llama_index.llms.openai.utils import GPT4_MODELS, ALL_AVAILABLE_MODELS
from llama_index.llms.anthropic.utils import ANTHROPIC_MODELS, CLAUDE_MODELS

from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer


CUSTOM_MODELS = {
    "gpt-4o": 128000,
    "gpt-4o-0513": 127000,
    "claude35_sonnet2": 200000,
    "claude-3-5-haiku": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
}

# 更新 GPT4_MODELS
GPT4_MODELS.update(CUSTOM_MODELS)
ALL_AVAILABLE_MODELS.update(CUSTOM_MODELS)

ANTHROPIC_MODELS.update(CUSTOM_MODELS)
CLAUDE_MODELS.update(CUSTOM_MODELS)

img_url="https://opencdn.dingtalk.net/vscode/puppeteer/cloud_runtime/18781d38-3175-48a8-9ab7-ca246cf444be_screenshot.png"
temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")

class ImageDecision(BaseModel):
    """Description of image"""
    describe: str


# 如果文件夹不存在就创建
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
temp_image_path = os.path.join(temp_dir, f"screenshot{hash(img_url)}.png")

img_response = requests.get(img_url, verify=False)

with open(temp_image_path, 'wb') as f:
    f.write(img_response.content)
history = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path=temp_image_path),
            TextBlock(text="What is in this image?"),
        ],
    ),
]
# llm = OpenAI(
#     model="gpt-4.1",
#     api_key="sk-omnC1J0vKTJKC9sm530cCeFa50A54a73Ab0922Ef869cBd29",
#     # api_base=os.getenv("OPENAI_API_BASE"),
#     api_base="https://www.gptapi.us/v1/",
#     base_url="https://api.gptapi.us",
# )
# response = llm.as_structured_llm(ImageDecision).chat(history)
# print(response.raw)

llm2 = OpenAI(
    model="gpt-4o-0513",
    api_key="a775cabefed422a58f16eb8d75d0d34b",
    # api_base=os.getenv("OPENAI_API_BASE"),
    api_base="https://idealab.alibaba-inc.com/api/openai/v1"
)
response2 = llm2.as_structured_llm(ImageDecision).chat(history)
print(response2.raw)