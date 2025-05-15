import os
from typing import List

import aiohttp
import requests
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

class Coordinate(BaseModel):
    app: str = Field(description="app名称", examples=["微信", "钉钉", "QQ", "支付宝"])
    x: str = Field(description="横坐标", examples=[100])
    y: str = Field(description="纵坐标", examples=[500])
class CoordinateList(BaseModel):
    coordinates: List[Coordinate] = Field(description="app的坐标列表")
# 结构化输出模型
class ActionDecision(BaseModel):
    apps: List[str] = Field(description="图片中有哪些app", examples=["微信", "钉钉", "QQ", "支付宝"])
    coordinate: Coordinate = Field(description="图片中app的坐标", examples=[{'date': '03-29'}, {'time': "8:30"}])
    is_ali_ding: bool = Field(description="是否安装了阿里钉")



img_url="https://opencdn.dingtalk.net/vscode/puppeteer/cloud_runtime/541af37b-6287-430c-93d1-3238b937da74_screenshot.png"
temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")

class ImageDecision(BaseModel):
    """Description of image"""
    describe: str


# 如果文件夹不存在就创建
# if not os.path.exists(temp_dir):
#     os.makedirs(temp_dir)
# temp_image_path = os.path.join(temp_dir, f"screen{hash(img_url)}.png")
#
# img_response = requests.get(img_url, verify=False)
#
# with open(temp_image_path, 'wb') as f:
#     f.write(img_response.content)
history = [
    ChatMessage(role="system", content="你会收到一张720x1280的图片，图片中包含了一个手机屏幕的截图，手机上有很多app图标。请你分析这张图片，告诉我图片中有哪些app，并基于坐标，告诉我每个app的中心坐标。"),
    ChatMessage(
        role="user",
        blocks=[
            ImageBlock(path='./tmp/screenshots/screen5490072818026924971.png'),
            # ImageBlock(path=temp_image_path),
            TextBlock(text="分析这张图片"),
        ],
    ),
]
# gpt-41-0414-global, gpt-4o-0806-global, qwen2.5-vl-72b-instruct✅, qwen-vl-max, gemini-2.5-pro-03-25
# llm4 = OpenAILike(
#     model="qwen2.5-vl-72b-instruct",
#     api_base=os.getenv("AI_STUDIO_BASE"),
#     api_key=os.getenv("AI_STUDIO_API_KEY"),
#     context_window=200000,
#     is_chat_model=True,
#     is_function_calling_model=False,
# ).as_structured_llm(CoordinateList)
# response = llm4.chat(history)
# decision: CoordinateList = response.raw
# print(decision)

# gpt-41-0414-global, gpt-4o-0806-global, qwen2.5-vl-72b-instruct✅, qwen-vl-max, gemini-2.5-pro-03-25
llm4 = OpenAILike(
    model="claude37_sonnet",
    api_base=os.getenv("AI_STUDIO_BASE"),
    api_key=os.getenv("AI_STUDIO_API_KEY"),
    context_window=200000,
    is_chat_model=True,
    is_function_calling_model=True,
)
response = llm4.chat([
    ChatMessage(role="system", content="你会收到一张720x1280的图片，图片中包含了一个手机屏幕的截图，手机上有很多app图标。请你分析这张图片，告诉我图片中有哪些app，并基于坐标，告诉我每个app的中心坐标。请按照json格式返回，例如：[{'apps': '微信', 'coordinate': {'x': 100, 'y': 200}, {'app': '钉钉', 'coordinate': {'x': 300, 'y': 400}}]，不要返回其他内容"),
    ChatMessage(
        role="user",
        blocks=[
            # ImageBlock(url=img_url),
            ImageBlock(path='./tmp/screenshots/screen5490072818026924971.png')
            # ImageBlock(path=temp_image_path),
        ],
    )])
print(str(response))

from llama_index.llms.anthropic import Anthropic

# llm = Anthropic(
#     model="claude-3-7-sonnet-latest",
#     api_key=os.getenv("ANTHROPIC_API_KEY")).as_structured_llm(ActionDecision)
# resp = llm.chat(history)
# desc: ActionDecision = resp.raw
# print(desc)

# from llama_index.llms.anthropic.utils import CLAUDE_MODELS
# CLAUDE_MODELS["claude-3-7-sonnet"] = 200000
#
# llm = Anthropic(model="claude-3-7-sonnet-20250219",
#                 base_url=os.getenv("GPTAPI_US_ANTHROPIC_BASE"),
#                 api_key=os.getenv("GPTAPI_US_KEY"),
#                 ).as_structured_llm(ActionDecision)
# resp = llm.chat(history)
# desc: ActionDecision = resp.raw
# print(desc)

# llm = OpenAILike(
#     model="gpt-4-1106-preview",
#     api_base=os.getenv("GPTAPI_US_OPENAI_BASE"),
#     api_key=os.getenv("GPTAPI_US_KEY"),
#     context_window=200000,
#     is_chat_model=True,
#     is_function_calling_model=True,
# ).as_structured_llm(ActionDecision)
#
# response = llm.chat(history)
# decision: ActionDecision = response.raw
# print(decision)