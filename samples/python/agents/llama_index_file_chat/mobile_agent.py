import asyncio
import json
import os
from enum import Enum
from typing import Dict, Any
from typing import List, Optional
import re

import time
from agents.llama_index_file_chat.agent_validation_utils import mcp_utils
import aiofiles
import aiohttp
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, TextBlock, ImageBlock
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent, step
from llama_index.llms.openai_like import OpenAILike
from llama_index.tools.mcp import BasicMCPClient
from pydantic import BaseModel, Field, model_validator
from agents.llama_index_file_chat.agent_validation_utils.data_utils import ActionStepChecker
from agents.llama_index_file_chat.agent_validation_utils.prompt_constants import ACTION_PROMPT
from llama_index.core.types import (
    BaseOutputParser
)
from typing import Any

from my_utils.logger_util import logger

ACTION_MCP_SERVER_URL = "http://118.178.191.176:8001/sse"

load_dotenv()

# 验证模式配置
AGENT_VALIDATION = os.getenv("AGENT_VALIDATION", "false").lower() == "true"

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
        description="执行操作要传入的JSON格式参数字符串，某些操作（如 screenshot、back）为空字符串",
        examples=[
            "",  # 无参数
            '{"x": 100, "y": 200}',  # click, long_press, double_click
            '{"x": 100, "y": 200, "text": "Hello, World!"}',  # type
            '{"start": [100, 200], "end": [300, 400]}',  # scroll
        ]
    )
    reasoning: str = Field(description="选择该操作的推理过程")

    @classmethod
    def from_json(cls, json_str: str) -> 'ActionDecision':
        """
        从JSON字符串创建ActionDecision对象
        """
        logger.info(f"kkk ActionDecision from_json json_str: {json_str!r}")
        return cls.model_validate_json(json_str)

    def to_json(self) -> str:
        """
        将ActionDecision对象转换为JSON字符串
        """
        return self.model_dump_json()

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
            if params and parsed_params:
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
    task: str  # 用户任务，例如"给小明发送你好的短信"


class ScreenshotEvent(Event):
    screenshot_url: str
    task: str


class ActionEvent(Event):
    action: str
    params: dict
    task: str
    action_description: str


class ActionResultEvent(Event):
    result: dict
    task: str


class TaskCompleteEvent(StopEvent):
    response: str
    history: List[Dict]

class CustomJSONParser(BaseOutputParser):
    def parse(self, output: str) -> Any:
        # 尝试修正常见的JSON格式错误
        logger.info(f"kkk CustomJSONParser parse output: {output!r}")
        output = repair_json_string(output)
        logger.info(f"kkk CustomJSONParser parse output after repair: {output!r}")
        # 解析JSON
        return output

MAX_STEPS = 10

# MobileInteractionAgent 工作流
class MobileInteractionAgent(Workflow):
    def __init__(self, timeout: Optional[float] = 300.0, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = OpenAILike(
            model="qwen2.5-vl-72b-instruct",
            api_base=os.getenv("AI_STUDIO_BASE"),
            api_key=os.getenv("AI_STUDIO_API_KEY"),
            context_window=131072,
            max_tokens=129024,
            is_chat_model=True,
            is_function_calling_model=False,
        )
        # .as_structured_llm(ActionDecision)
        # TODO 使用structured_llm尝试使用自定义output_parser

        # self.llm = OpenAILike(
        #     model="claude37_sonnet",
        #     api_base=os.getenv("AI_STUDIO_BASE"),
        #     api_key=os.getenv("AI_STUDIO_API_KEY"),
        #     context_window=200000,
        #     is_chat_model=True,
        #     is_function_calling_model=True,
        # )
        self.client = BasicMCPClient(ACTION_MCP_SERVER_URL)

        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)

        self.system_prompt = PromptTemplate(ACTION_PROMPT)

        self.action_step_checker = None

        self.running_steps = 0

        self.mcp_client = mcp_utils.MCPClient()

    async def get_screenshot(self, instance_id:str) -> str:
        try:
            response = await self.client.call_tool("get_screenshot", {"instance_id": instance_id})
            return response.content[0].text
        except Exception as e:
            logger.warning(f"First screenshot attempt failed: {str(e)}. Retrying once...")
            response = await self.client.call_tool("get_screenshot", {"instance_id": instance_id})
            return response.content[0].text

    # execute_action 工具函数
    async def execute_action(self,instance_id:str, action: str, params: dict) -> Dict:
        try:
            response = await self.client.call_tool("execute_action", {"instance_id": instance_id, "action": action, "params": params})
            return json.loads(response.content[0].text)
        except Exception as e:
            logger.warning(f"First execute_action attempt failed: {str(e)}. Retrying once...")
            response = await self.client.call_tool("execute_action", {"instance_id": instance_id, "action": action, "params": params})
            return json.loads(response.content[0].text)

    @step
    async def start(self, ctx: Context, ev: TaskEvent) -> ScreenshotEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Starting task: {ev.task}"))
        task_id = hash(ev.task+str(time.time()))

        await ctx.set("task", ev.task)
        await ctx.set("history", [])  # 初始化执行历史
        await ctx.set("task_id", task_id)

        metadata = await ctx.get("metadata", default={})
        instance_id = metadata["instanceId"]
        if not instance_id:
            logger.error("No instanceId provided in TaskEvent")
            raise ValueError("No instanceId provided in TaskEvent")
        
        self.action_step_checker = ActionStepChecker(task=ev.task, task_id=task_id)

        screenshot_url = await self.get_screenshot(instance_id)
        ctx.write_event_to_stream(LogEvent(msg=f"Retrieved screenshot: {screenshot_url}"))
        return ScreenshotEvent(screenshot_url=screenshot_url, task=ev.task)

    @step
    async def analyze_screenshot(self, ctx: Context, ev: ScreenshotEvent) -> ActionEvent | TaskCompleteEvent | ScreenshotEvent:
        task = await ctx.get("task")
        history = await ctx.get("history", default=[])
        history_summary = "\n".join(
            [f"Step {i + 1}: {h['action']} with {h['params']} -> {h['result']['message']} -> action_description:{h['action_description']}" for i, h in
             enumerate(history)])

        if not isinstance(self.llm, StructuredLLM):
            prompt = self.system_prompt.format(
                task=task,
                history=history_summary or "No actions taken yet",
                output_example="输出格式需要符合以下JSON格式：\n{'action': 'click', 'params': {'x': 100, 'y': 200}, 'reasoning': '使用这个操作的原因，不要使用除文字以外的特殊字符'} \n 确保这个json是合法的，不需要有其他的内容",
            )
        else:
            prompt = self.system_prompt.format(
                task=task,
                history=history_summary or "No actions taken yet",
                output_example="输出格式需要符合以下JSON格式：\n{'action': 'click', 'params': {'x': 100, 'y': 200}, 'reasoning': '使用这个操作的原因，不要使用除文字以外的特殊字符'} \n 确保这个json是合法的，不需要有其他的内容"
            )

        temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")
        # 如果文件夹不存在就创建
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_image_path = os.path.join(temp_dir, f"{hash(ev.screenshot_url)}.png")

        async with aiohttp.ClientSession() as session:
            async with session.get(ev.screenshot_url, ssl=False) as response:
                img_content = await response.read()
                async with aiofiles.open(temp_image_path, 'wb') as f:
                    await f.write(img_content)

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

        try:
            # 调用 LLM
            response = await self.llm.achat(messages)
            logger.info(f"kkk LLM response raw: {response.raw!r}")
            decision: ActionDecision
            if not isinstance(self.llm, StructuredLLM):
                try:
                    content = repair_json_string(response.message.content)
                    temp = json.loads(content)
                    # 确保 params 是字符串格式
                    if isinstance(temp.get('params'), dict):
                        temp['params'] = json.dumps(temp['params'])
                    decision = ActionDecision.from_json(json.dumps(temp))
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse LLM response. Raw response: {response.message.content!r}")
                    logger.error(f"Parse error details: {str(e)}")
                    return ScreenshotEvent(screenshot_url=ev.screenshot_url, task=ev.task)
            else:
                decision = response.raw

            ctx.write_event_to_stream(
                LogEvent(msg=f"Decision: {decision.action} with params {decision.params}, Reasoning: {decision.reasoning}"))

            if AGENT_VALIDATION:
                # 保存过程图片
                self.action_step_checker.save_process_screenshot(temp_image_path)
            else:   
                # 函数结束后可以选择删除临时文件
                os.remove(temp_image_path)
            # 更新对话历史
            self.memory.put(response.message)

            if decision.action == ActionType.DONE:
                return TaskCompleteEvent(response=f"Task completed: {decision.reasoning}", history=history)

            return ActionEvent(action=decision.action, params=decision.get_parsed_params(), task=task, action_description=decision.reasoning)

        except Exception as e:
            logger.error(f"Error in analyze_screenshot: {str(e)}")
            logger.error(f"Full error details:", exc_info=True)
            if hasattr(e, '__dict__'):
                logger.error(f"Error attributes: {e.__dict__}")
            return ScreenshotEvent(screenshot_url=ev.screenshot_url, task=ev.task)

    @step
    async def execute_action_step(self, ctx: Context, ev: ActionEvent) -> ActionResultEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Executing action: {ev.action} with params {ev.params}"))
        metadata = await ctx.get("metadata", default={})
        instance_id = metadata["instanceId"]
        result = await self.execute_action(instance_id, ev.action, ev.params)
        self.running_steps += 1
        ctx.write_event_to_stream(LogEvent(msg=f"Action result: {result['message']}"))

        # 更新执行历史
        history = await ctx.get("history", default=[])
        history.append({"action": ev.action, "params": ev.params, "result": result, "action_description": ev.action_description})
        await ctx.set("history", history)

        return ActionResultEvent(result=result, task=ev.task)

    @step
    async def check_completion(self, ctx: Context, ev: ActionResultEvent) -> ScreenshotEvent | TaskCompleteEvent:
        history = await ctx.get("history", default=[])
        if not ev.result["success"]:
            return TaskCompleteEvent(response=f"Task failed: {ev.result['message']}", history=history)
        
        if self.running_steps >= MAX_STEPS:
            return TaskCompleteEvent(response=f"Task completed: over max steps", history=history)

        # 截图前延迟一段时间，以确保操作完成
        await asyncio.sleep(2)
        # 获取新的截图，继续任务
        metadata = await ctx.get("metadata", default={})
        instance_id = metadata["instanceId"]
        screenshot_url = await self.get_screenshot(instance_id)
        ctx.write_event_to_stream(LogEvent(msg=f"Retrieved new screenshot: {screenshot_url}"))
        return ScreenshotEvent(screenshot_url=screenshot_url, task=ev.task)
    
    async def execute_back_home(self, instance_id:str):
        await self.execute_action(instance_id, "home", {})

def repair_json_string(json_str: str) -> Optional[str]:
    """
    修复常见的JSON格式问题
    1. 处理缺少引号的键
    2. 处理多余的逗号
    3. 处理单引号
    4. 处理布尔值
    5. 处理数组格式问题
    6. 处理括号匹配问题
    """
    original_json_str = json_str
    try:
        # 尝试直接解析，如果成功则返回
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    # 1. 处理缺少引号的键
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    
    # 2. 处理单引号
    json_str = json_str.replace("\\'", '')
    
    # 3. 处理布尔值
    json_str = re.sub(r':\s*true\s*([,}])', r': true\1', json_str)
    json_str = re.sub(r':\s*false\s*([,}])', r': false\1', json_str)
    
    # 4. 处理数组格式问题
    json_str = re.sub(r'\[\s*([^]]*?)\s*\]', lambda m: '[' + m.group(1).strip() + ']', json_str)
    
    # 5. 处理多余的逗号
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # 6. 处理括号匹配问题
    # 处理对象中的中括号，但排除正常的数组格式
    # 只处理包含键值对的中括号
    json_str = re.sub(r'([{,])\s*"([^"]+)"\s*:\s*\[([^]]*?)([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^]]+)\]', r'\1"\2": {\3"\4": \5}', json_str)
    # 处理数组中的大括号
    json_str = re.sub(r'\[\s*{([^}]+)}\s*\]', r'[{\1}]', json_str)
    
    # 7. 处理错误的中括号闭合
    # 处理对象中错误使用中括号闭合的情况
    json_str = re.sub(r'([{,])\s*"([^"]+)"\s*:\s*([^,}]+)\]', r'\1"\2": \3}', json_str)
    
    try:
        # 验证修复后的JSON是否有效
        json.loads(json_str)
        logger.info(f"repair JSON: from {original_json_str} to {json_str}")
        return json_str
    except json.JSONDecodeError as e:
        logger.error(f"Failed to repair JSON: {json_str}")
        logger.error(f"Repair error: {str(e)}")
        return None

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

async def main_validation(json_name:str):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "agent_validation_utils", "test_datas", f"{json_name}.json")
    with open(json_path, "r") as f:
        diff_level_array = json.load(f)
    for diff_jo in diff_level_array:
        await single_run(diff_jo)
    ActionStepChecker.gen_stats_report()

async def single_run(diff_jo: dict):
    '''
    {
    "task_id": "modify_status_to_empty",
    "instruction": "修改我的状态为无状态",
    "diff_level": "1",
    "rubrics": [
      "中间步骤出现「我的状态」页面",
      "「我的状态」页面中，「无状态」处于选中状态",
      "如果最后在首页，那么左上角的头像中右下角显示电脑或者手机设备的图标，而不是一个emoji",
      "如果最后是在「我的」页面，那么在资料卡片中显示 (+状态...),这表示当前是无状态"
    ],
    "goldStep": 5,
    "reset": "set_work_status_to_vacation"
  }
    '''

    """测试 MobileInteractionAgent"""
    agent = MobileInteractionAgent()
    ctx = Context(agent)
    instance_id = "acp-9nfh0qgrttkkfi6kc"
    await ctx.set("metadata", {"instanceId": instance_id}) # 验证模式下使用的设备

    try:
        # 从diff_jo中获取task并使用校验数据
        task = diff_jo["instruction"]
        await ctx.set("diff_jo", diff_jo)
        handler = agent.run(start_event=TaskEvent(task=task), ctx=ctx)

        async for event in handler.stream_events():
            if isinstance(event, LogEvent):
                print(f"Log: {event.msg}")

        result: TaskCompleteEvent = await handler
        print(f"Final Response: {result.response}")
        print("Execution History:")
        for i, step in enumerate(result.history):
            print(f"Step {i + 1}: {step['action']} with {step['params']} -> {step['result']['message']}")

        # 检查最终结果是否符合预期
        await agent.action_step_checker.check_final_result(diff_jo, result.history)

        await agent.execute_back_home(instance_id)

        # 等待1.5秒
        await asyncio.sleep(1.5)

    except Exception as e:
        logger.error(f"Error during execution: {e}")

if __name__ == "__main__":
    if not AGENT_VALIDATION:
        asyncio.run(main())
    else:
        asyncio.run(main_validation('diff_level_1'))