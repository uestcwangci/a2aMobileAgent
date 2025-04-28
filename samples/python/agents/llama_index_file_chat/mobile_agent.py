import asyncio
import json
from typing import Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field
from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent, step
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock
import os
from aliyun.instance_manager import InstanceManager
from my_utils.logger_util import logger
from dotenv import load_dotenv
import ast
import tempfile
import requests
import os

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
}

# 更新 GPT4_MODELS
GPT4_MODELS.update(CUSTOM_MODELS)
ALL_AVAILABLE_MODELS.update(CUSTOM_MODELS)

ANTHROPIC_MODELS.update(CUSTOM_MODELS)
CLAUDE_MODELS.update(CUSTOM_MODELS)

load_dotenv()

instance_manager = InstanceManager()

async def get_screenshot(ctx: Context) -> str:
    metadata = await ctx.get("metadata", default={})
    instance_id = metadata["instanceId"]
    appium_action = instance_manager.get_or_create_client(instance_id)
    return appium_action.execute("screenshot")["screenshot"]


# execute_action 工具函数
async def execute_action(ctx: Context, action: str, params: dict) -> Dict:
    metadata = await ctx.get("metadata", default={})
    instance_id = metadata["instanceId"]
    logger.info(f"instance_id: {instance_id}")
    appium_action = instance_manager.get_or_create_client(instance_id)
    result = appium_action.execute(action, params)
    return result


# 事件定义
class LogEvent(Event):
    msg: str


class TaskEvent(StartEvent):
    task: str  # 用户任务，例如“给小明发送你好的短信”


class ScreenshotEvent(Event):
    screenshot_url: str
    task: str


class ActionEvent(Event):
    action: str
    params: dict
    task: str


class ActionResultEvent(Event):
    result: dict
    task: str


class TaskCompleteEvent(StopEvent):
    response: str
    history: List[Dict]


# 结构化输出模型
class ActionDecision(BaseModel):
    action: str = Field(description="要执行的操作（click, type, long_press 等）")
    params: str = Field(description="操作参数 例如 {'app_name': 'DingTalk'} 或 {'x': int, 'y': int} 或 {'x': int, 'y': int, 'text': str}")
    reasoning: str = Field( description="选择该操作的推理过程")
    is_complete: bool = Field(description="任务是否已完成")


# MobileInteractionAgent 工作流
class MobileInteractionAgent(Workflow):
    def __init__(self, timeout: Optional[float] = 300.0, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = OpenAI(
            model="gpt-4.5-preview",
            api_key="sk-omnC1J0vKTJKC9sm530cCeFa50A54a73Ab0922Ef869cBd29",
            # api_base=os.getenv("OPENAI_API_BASE"),
            api_base="https://www.gptapi.us/v1/",
            # base_url=os.getenv("ANTHROPIC_BASE_URL"),
        ).as_structured_llm(ActionDecision)
        # self.llm = Anthropic(
        #     model="claude-3-5-haiku",
        #     api_key="sk-omnC1J0vKTJKC9sm530cCeFa50A54a73Ab0922Ef869cBd29",
        #     base_url="https://api.gptapi.us",
        # ).as_structured_llm(ActionDecision)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        self.system_prompt = """\
You are a MobileInteractionAgent that operates a mobile device in a human-like manner to complete user tasks.
Your goal is to analyze the current screenshot, task, and execution history to decide the next action.

**Screen Information**:
- The screenshot dimensions are 720 pixels (width) × 1280 pixels (height).
- The coordinate origin (0, 0) is at the top-left corner of the screen.
- The x-axis increases to the right (x ranges from 0 to 719).
- The y-axis increases downward (y ranges from 0 to 1279).
- For actions requiring coordinates (e.g., click, type, long_press, double_click, scroll), ensure the x and y values are within these ranges.

Available actions:
- click: Click at coordinates {{"x": int, "y": int}}. Description: Click at specified coordinates, e.g., {{"x": 100, "y": 100}}.
- type: Type text at coordinates {{"x": int, "y": int, "text": str}}. Description: Type text at specified coordinates, e.g., {{"x": 100, "y": 100, "text": "Hello, World!"}}.
- long_press: Long press at coordinates {{"x": int, "y": int}}. Description: Long press at specified coordinates, e.g., {{"x": 200, "y": 200}}.
- screenshot: Take a new screenshot. Description: If the screenshot cannot be recognized, take a new one. Parameters: {{}}.
- double_click: Double click at coordinates {{"x": int, "y": int}}. Description: Double click at specified coordinates, e.g., {{"x": 300, "y": 300}}.
- back: Go back. Description: Navigate back. Parameters: {{}}.
- scroll: Scroll with coordinates {{"start": [int, int], "end": [int, int]}}. Description: Scroll from start to end coordinates, e.g., {{"start": [400, 400], "end": [500, 500]}}.
- launch_app: Launch an app {{"app_name": str}}. Description: Launch the specified app, e.g., {{"app_name": "DingTalk"}}.
- press_enter or press_search: Send enter or search keycode. Description: Simulate pressing the enter or search key. Parameters: {{}}.
- done: Finish the task. Description: Indicate the task is complete. Parameters: {{}}.

Rules:
1. Analyze the screenshot URL to understand the current screen context.
2. Consider the task and execution history to decide the next action.
3. Provide a clear reasoning for your action choice.
4. Use the exact parameter format specified for each action.
5. For coordinate-based actions, ensure x is between 0 and 719, and y is between 0 and 1279.
6. You must set action, params, reasoning and is_complete in the response. Even if the empty string or empty dictionary.
7. If the task is started, select the 'launch_app' action with the app_name as params, defaulting to {{"app_name":"DingTalk"}}.
8. If the task is complete, select the 'done' action with empty parameters and set is_complete to True.
9. If no action is needed or the task cannot proceed, explain why, select 'done', and set is_complete to True.
10. If the screenshot is unclear or cannot be recognized, consider using the 'screenshot' action to refresh.

Task: {task}
Execution History: {history}
Current Screenshot: {screenshot_url}
"""

    @step
    async def start(self, ctx: Context, ev: TaskEvent) -> ScreenshotEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Starting task: {ev.task}"))
        await ctx.set("task", ev.task)
        await ctx.set("history", [])  # 初始化执行历史
        metadata = await ctx.get("metadata", default={})
        if not metadata["instanceId"]:
            logger.error("No instanceId provided in TaskEvent")
            raise ValueError("No instanceId provided in TaskEvent")

        screenshot_url = await get_screenshot(ctx)
        ctx.write_event_to_stream(LogEvent(msg=f"Retrieved screenshot: {screenshot_url}"))
        return ScreenshotEvent(screenshot_url=screenshot_url, task=ev.task)

    @step
    async def analyze_screenshot(self, ctx: Context, ev: ScreenshotEvent) -> ActionEvent | TaskCompleteEvent:
        task = await ctx.get("task")
        history = await ctx.get("history", default=[])
        history_summary = "\n".join(
            [f"Step {i + 1}: {h['action']} with {h['params']} -> {h['result']['message']}" for i, h in
             enumerate(history)])

        # 构造 LLM 输入
        prompt = self.system_prompt.format(
            task=task,
            history=history_summary or "No actions taken yet",
            screenshot_url=ev.screenshot_url
        )

        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, f"screenshot{hash(ev.screenshot_url)}.png")

        img_response = requests.get(ev.screenshot_url, verify=False)

        with open(temp_image_path, 'wb') as f:
            f.write(img_response.content)
        # # 异步下载图片
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(ev.screenshot_url) as response:
        #         img_content = await response.read()
        #         with open(temp_image_path, 'wb') as f:
        #             f.write(img_content)

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", blocks=[
                ImageBlock(path=temp_image_path),
                TextBlock(text=f"Analyze the screenshot and decide the next action for task: {task}")
            ])
        ]

        # 函数结束后可以选择删除临时文件
        # os.remove(temp_image_path)

        # 获取对话历史
        chat_history = await self.memory.aget_all()
        messages = chat_history + messages
        ctx.write_event_to_stream(LogEvent(msg="Analyzing screenshot and task..."))

        # 调用 LLM
        response = await self.llm.achat(messages)
        decision: ActionDecision = response.raw
        ctx.write_event_to_stream(
            LogEvent(msg=f"Decision: {decision.action} with params {decision.params}, Reasoning: {decision.reasoning}"))

        # 更新对话历史
        self.memory.put(response.message)

        if decision.is_complete:
            return TaskCompleteEvent(response=f"Task completed: {decision.reasoning}", history=history)

        return ActionEvent(action=decision.action, params=ast.literal_eval(decision.params), task=task)

    @step
    async def execute_action_step(self, ctx: Context, ev: ActionEvent) -> ActionResultEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Executing action: {ev.action} with params {ev.params}"))
        result = await execute_action(ctx, ev.action, ev.params)
        ctx.write_event_to_stream(LogEvent(msg=f"Action result: {result['message']}"))

        # 更新执行历史
        history = await ctx.get("history", default=[])
        history.append({"action": ev.action, "params": ev.params, "result": result})
        await ctx.set("history", history)

        return ActionResultEvent(result=result, task=ev.task)

    @step
    async def check_completion(self, ctx: Context, ev: ActionResultEvent) -> ScreenshotEvent | TaskCompleteEvent:
        if not ev.result["success"]:
            history = await ctx.get("history", default=[])
            return TaskCompleteEvent(response=f"Task failed: {ev.result['message']}", history=history)

        # 获取新的截图，继续任务
        screenshot_url = await get_screenshot(ctx)
        ctx.write_event_to_stream(LogEvent(msg=f"Retrieved new screenshot: {screenshot_url}"))
        return ScreenshotEvent(screenshot_url=screenshot_url, task=ev.task)


async def main():
    """测试 MobileInteractionAgent"""
    agent = MobileInteractionAgent()
    ctx = Context(agent)
    await ctx.set("metadata", {"instanceId": "acp-1hs2j2n7fjpzyn79a"})

    # 测试任务
    task = "Send a message 'Hello' to 零封 via DingTalk"
    handler = agent.run(start_event=TaskEvent(task=task), ctx=ctx)

    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            print(f"Log: {event.msg}")

    result: TaskCompleteEvent = await handler
    print(f"Final Response: {result.response}")
    print("Execution History:")
    for i, step in enumerate(result.history):
        print(f"Step {i + 1}: {step['action']} with {step['params']} -> {step['result']['message']}")


if __name__ == "__main__":
    asyncio.run(main())