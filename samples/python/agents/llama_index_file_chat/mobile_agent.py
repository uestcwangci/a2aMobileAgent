import asyncio
import json
from typing import Dict, List, Optional

import aiohttp
import json
from typing import Dict, Any, List
from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent, step
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
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel, Field, model_validator
from typing import List, Union, Optional
from enum import Enum
from llama_index.llms.anthropic import Anthropic


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


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    LONG_PRESS = "long_press"
    SCREENSHOT = "screenshot"
    DOUBLE_CLICK = "double_click"
    BACK = "back"
    SCROLL = "scroll"
    PRESS_ENTER = "press_enter"
    PRESS_SEARCH = "press_search"
    DONE = "done"

class ActionDecision(BaseModel):
    action: ActionType = Field(description="要执行的操作")
    params: str = Field(
        title="操作参数",
        description="执行操作要传入的JSON格式参数字符串，某些操作（如 screenshot、back、done）为空字符串",
        examples=[
            "",  # 无参数
            '{"x": 100, "y": 200}',  # click, long_press, double_click
            '{"x": 100, "y": 200, "text": "Hello, World!"}',  # type
            '{"start": [100, 200], "end": [300, 400]}',  # scroll
        ]
    )
    reasoning: str = Field(description="选择该操作的推理过程")
    is_complete: bool = Field(description="任务是否已完成")

    @model_validator(mode="after")
    def validate_action_params(cls, values):
        action = values.action
        params = values.params

        # 定义 action 和 params 的对应关系
        no_params_actions = [ActionType.SCREENSHOT, ActionType.BACK, ActionType.PRESS_ENTER, ActionType.PRESS_SEARCH, ActionType.DONE]
        coordinate_actions = [ActionType.CLICK, ActionType.LONG_PRESS, ActionType.DOUBLE_CLICK]
        type_action = [ActionType.TYPE]
        scroll_actions = [ActionType.SCROLL]

        # 验证 params 是否为有效的 JSON 字符串
        parsed_params: Dict[str, Any] = {}
        if params:
            try:
                parsed_params = json.loads(params)
            except json.JSONDecodeError:
                raise ValueError(f"Params must be a valid JSON string, got: {params}")

        # 验证 action 和 params 的匹配
        if action in no_params_actions:
            if params:
                raise ValueError(f"Action '{action}' does not require params, but params were provided: {params}")
        elif action in coordinate_actions:
            if not parsed_params.get("x") or not parsed_params.get("y"):
                raise ValueError(f"Action '{action}' requires 'x' and 'y' in params, got: {parsed_params}")
        elif action in type_action:
            if not parsed_params.get("x") or not parsed_params.get("y") or not parsed_params.get("text"):
                raise ValueError(f"Action 'type' requires 'x', 'y', and non-empty 'text' in params, got: {parsed_params}")
        elif action in scroll_actions:
            if not parsed_params.get("start") or not parsed_params.get("end"):
                raise ValueError(f"Action 'scroll' requires 'start' and 'end' in params, got: {parsed_params}")
            if not (isinstance(parsed_params["start"], list) and isinstance(parsed_params["end"], list)):
                raise ValueError(f"Action 'scroll' requires 'start' and 'end' to be lists, got: {parsed_params}")

        return values

    def get_parsed_params(self) -> Dict[str, Any]:
        """解析 params 字符串为 Python 字典"""
        if not self.params:
            return {}
        try:
            return json.loads(self.params)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse params as JSON: {self.params}, error: {str(e)}")

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


# MobileInteractionAgent 工作流
class MobileInteractionAgent(Workflow):
    def __init__(self, timeout: Optional[float] = 300.0, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = OpenAILike(
            model="gpt-4o-0806",
            api_base=os.getenv("AI_STUDIO_BASE"),
            api_key=os.getenv("AI_STUDIO_API_KEY"),
            context_window=200000,
            is_chat_model=True,
            is_function_calling_model=True,
        ).as_structured_llm(ActionDecision)

        # self.llm = Anthropic(
        #     model="claude-3-7-sonnet-20250219",
        #     base_url=os.getenv("GPTAPI_US_ANTHROPIC_BASE"),
        #     api_key=os.getenv("GPTAPI_US_KEY")
        # ).as_structured_llm(ActionDecision)

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

        self.system_prompt = PromptTemplate("""
你是一个以人性化方式操作移动设备来完成用户任务的MobileInteractionAgent。你的目标是分析当前截图、任务和执行历史，以决定下一步操作。

屏幕信息：
- 截图尺寸为720像素（宽）× 1280像素（高）
- 坐标原点(0, 0)位于屏幕左上角
- x轴向右递增（x范围从0到719）
- y轴向下递增（y范围从0到1279）
- 对于需要坐标的操作（如点击、输入、长按、双击、滑动），请确保x和y值在这些范围内

可用操作：
- click：在坐标处点击 {"x": int, "y": int}。描述：在指定坐标处点击，例如：{"x": 100, "y": 100}。
- type：在坐标处输入文本 {"x": int, "y": int, "text": str}。描述：在指定坐标处输入文本，例如：{"x": 100, "y": 100, "text": "你好，世界！"}。
- long_press：在坐标处长按 {"x": int, "y": int}。描述：在指定坐标处长按，例如：{"x": 200, "y": 200}。
- screenshot：拍摄新的截图。描述：如果无法识别截图，则拍摄新的截图。参数：{}。
- double_click：在坐标处双击 {"x": int, "y": int}。描述：在指定坐标处双击，例如：{"x": 300, "y": 300}。
- back：返回。描述：导航返回。参数：{}。
- scroll：滑动 {"start": [int, int], "end": [int, int]}。描述：从起始坐标滑动到结束坐标，例如：{"start": [400, 400], "end": [500, 500]}。
- press_enter或press_search：发送回车或搜索按键码。描述：模拟按下回车或搜索键。参数：{}。
- done：完成任务。描述：表示任务已完成。参数：{}。

规则：
- 分析截图URL以理解当前屏幕上下文
- 如果没有指定App，默认使用钉钉
- 考虑任务和执行历史来决定下一步操作
- 为您的操作选择提供清晰的理由说明
- 使用每个操作指定的准确参数格式
- 对于基于坐标的操作，确保x在0到719之间，y在0到1279之间
- 您必须在响应中设置action、params、reasoning和is_complete。即使是空字符串或空字典
- 如果任务完成，选择'done'操作，使用空参数并将is_complete设置为True
- 如果截图不清晰或无法识别，考虑使用'screenshot'操作重新获取截图

任务：{task}
执行历史：{history}
""")
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
        )

        temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")
        # 如果文件夹不存在就创建
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_image_path = os.path.join(temp_dir, f"{hash(ev.screenshot_url)}.png")

        img_response = requests.get(ev.screenshot_url, verify=False)

        with open(temp_image_path, 'wb') as f:
            f.write(img_response.content)
        # 异步下载图片
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(ev.screenshot_url) as response:
        #         img_content = await response.read()
        #         with open(temp_image_path, 'wb') as f:
        #             f.write(img_content)

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(role="user", blocks=[
                ImageBlock(path=temp_image_path),
                TextBlock(text=f"分析手机截图，并且决定下一步动作来完成任务：{task}")
            ])
        ]



        # 获取对话历史
        chat_history = await self.memory.aget_all()
        messages = chat_history + messages
        ctx.write_event_to_stream(LogEvent(msg="Analyzing screenshot and task..."))

        # 调用 LLM
        response = await self.llm.achat(messages)
        decision: ActionDecision = response.raw
        ctx.write_event_to_stream(
            LogEvent(msg=f"Decision: {decision.action} with params {decision.params}, Reasoning: {decision.reasoning}"))

        # 函数结束后可以选择删除临时文件
        os.remove(temp_image_path)
        # 更新对话历史
        self.memory.put(response.message)

        if decision.is_complete:
            return TaskCompleteEvent(response=f"Task completed: {decision.reasoning}", history=history)

        return ActionEvent(action=decision.action, params=decision.get_parsed_params(), task=task)

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
    task = "给零封发消息，说你好"
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