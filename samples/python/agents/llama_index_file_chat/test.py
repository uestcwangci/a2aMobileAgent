import os
from typing import List

import requests
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

class DateTime(BaseModel):
    date: str = Field(description="日期", examples=["03-29"])
    time: str = Field(description="时间", examples=["8:30"])
# 结构化输出模型
class ActionDecision(BaseModel):
    apps: List[str] = Field(description="图片中有哪些app", examples=["微信", "钉钉", "QQ", "支付宝"])
    datetime: DateTime = Field(description="图片显示的时间", examples=[{'date': '03-29'}, {'time': "8:30"}])
    is_ali_ding: bool = Field(description="是否安装了阿里钉")



img_url="https://opencdn.dingtalk.net/vscode/puppeteer/cloud_runtime/18781d38-3175-48a8-9ab7-ca246cf444be_screenshot.png"
temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")

class ImageDecision(BaseModel):
    """Description of image"""
    describe: str


# 如果文件夹不存在就创建
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
temp_image_path = os.path.join(temp_dir, f"screen{hash(img_url)}.png")

img_response = requests.get(img_url, verify=False)

with open(temp_image_path, 'wb') as f:
    f.write(img_response.content)
history = [
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(url=img_url),
            TextBlock(text="What is in this image?"),
        ],
    ),
]

# llm4 = OpenAILike(
#     model="gpt-4o-0806-global",
#     api_base="https://idealab.alibaba-inc.com/api/openai/v1",
#     api_key="",
#     context_window=200000,
#     is_chat_model=True,
#     is_function_calling_model=True,
# ).as_structured_llm(ActionDecision)
#
# response = llm4.chat(history)
# decision: ActionDecision = response.raw
# print(decision)

from llama_index.llms.anthropic import Anthropic

llm = Anthropic(
    model="claude35_sonnet2",
    api_key=os.getenv("ANTHROPIC_API_KEY")).as_structured_llm(ActionDecision)
resp = llm.chat(history)
desc: ActionDecision = resp.raw
print(desc)

llm = Anthropic(model="claude-3-7-sonnet-20250219",
                base_url=os.getenv("GPTAPI_US_ANTHROPIC_BASE"),
                api_key=os.getenv("GPTAPI_US_KEY"),
                ).as_structured_llm(ActionDecision)
resp = llm.chat(history)
desc: ActionDecision = resp.raw
print(desc)