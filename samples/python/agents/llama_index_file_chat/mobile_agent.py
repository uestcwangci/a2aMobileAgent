import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from llama_index.core.workflow import Workflow, Context, Event, StartEvent, StopEvent, step
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.memory import ChatMemoryBuffer
import os


# 模拟 Device.get_screenshot() 方法
class Device:
    @staticmethod
    def get_screenshot() -> str:
        """模拟获取手机实时截图的 HTTP 地址"""
        return "http://example.com/screenshot.png"  # 替换为实际截图 URL


# execute_action 工具函数（根据你的描述实现）
def execute_action(action: str, params: Dict) -> Dict:
    """
    执行手机操作。
    :param action: 操作类型（click, type, long_press 等）
    :param params: 操作参数（根据 action 不同而变化）
    :return: 操作结果
    """
    valid_actions = ["click", "type", "long_press"]
    if action not in valid_actions:
        return {"success": False, "message": f"Invalid action: {action}"}

    if action in ["click", "long_press"] and not ("x" in params and "y" in params):
        return {"success": False, "message": f"{action} requires x, y coordinates"}
    if action == "type" and not ("x" in params and "y" in params and "text" in params):
        return {"success": False, "message": "type requires x, y coordinates and text"}

    # 模拟执行操作（实际中应调用 Appium 或其他驱动）
    return {"success": True, "message": f"Executed {action} with params {params}"}


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
    params: Dict
    task: str


class ActionResultEvent(Event):
    result: Dict
    task: str


class TaskCompleteEvent(StopEvent):
    response: str
    history: List[Dict]


# 结构化输出模型
class ActionDecision(BaseModel):
    action: str = Field(description="要执行的操作（click, type, long_press 等）")
    params: Dict = Field(description="操作参数，例如 {'x': int, 'y': int} 或 {'x': int, 'y': int, 'text': str}")
    reasoning: str = Field(description="选择该操作的推理过程")
    is_complete: bool = Field(description="任务是否已完成")


# MobileInteractionAgent 工作流
class MobileInteractionAgent(Workflow):
    def __init__(self, timeout: Optional[float] = 300.0, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")).as_structured_llm(
            ActionDecision)
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
        self.system_prompt = """\
You are a MobileInteractionAgent that operates a mobile device in a human-like manner to complete user tasks.
Your goal is to analyze the current screenshot, task, and execution history to decide the next action.

Available actions:
- click: Click at coordinates {"x": int, "y": int}
- type: Type text at coordinates {"x": int, "y": int, "text": str}
- long_press: Long press at coordinates {"x": int, "y": int}

Rules:
1. Analyze the screenshot URL to understand the current screen context.
2. Consider the task and execution history to decide the next action.
3. Provide a clear reasoning for your action choice.
4. If the task is complete, set is_complete to True.
5. If no action is needed or the task cannot proceed, explain why and set is_complete to True.

Task: {task}
Execution History: {history}
Current Screenshot: {screenshot_url}
"""

    @step
    async def start(self, ctx: Context, ev: TaskEvent) -> ScreenshotEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Starting task: {ev.task}"))
        await ctx.set("task", ev.task)
        await ctx.set("history", [])  # 初始化执行历史
        screenshot_url = Device.get_screenshot()
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
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user",
             "content": f"Analyze the screenshot at {ev.screenshot_url} and decide the next action for task: {task}"}
        ]

        # 获取对话历史
        chat_history = await self.memory.get_all()
        messages = chat_history + messages
        ctx.write_event_to_stream(LogEvent(msg="Analyzing screenshot and task..."))

        # 调用 LLM
        response = await self.llm.achat(messages)
        decision: ActionDecision = response.raw
        ctx.write_event_to_stream(
            LogEvent(msg=f"Decision: {decision.action} with params {decision.params}, Reasoning: {decision.reasoning}"))

        # 更新对话历史
        self.memory.put({"role": "assistant", "content": str(decision)})

        if decision.is_complete:
            return TaskCompleteEvent(response=f"Task completed: {decision.reasoning}", history=history)

        return ActionEvent(action=decision.action, params=decision.params, task=task)

    @step
    async def execute_action_step(self, ctx: Context, ev: ActionEvent) -> ActionResultEvent:
        ctx.write_event_to_stream(LogEvent(msg=f"Executing action: {ev.action} with params {ev.params}"))
        result = execute_action(ev.action, ev.params)
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
        screenshot_url = Device.get_screenshot()
        ctx.write_event_to_stream(LogEvent(msg=f"Retrieved new screenshot: {screenshot_url}"))
        return ScreenshotEvent(screenshot_url=screenshot_url, task=ev.task)


async def main():
    """测试 MobileInteractionAgent"""
    agent = MobileInteractionAgent()
    ctx = Context(agent)

    # 测试任务
    task = "Send a message 'Hello' to XiaoMing via SMS"
    handler = agent.run(start_event=TaskEvent(task=task), ctx=ctx)

    async for event in handler:
        if isinstance(event, LogEvent):
            print(f"Log: {event.msg}")

    result: TaskCompleteEvent = await handler
    print(f"Final Response: {result.response}")
    print("Execution History:")
    for i, step in enumerate(result.history):
        print(f"Step {i + 1}: {step['action']} with {step['params']} -> {step['result']['message']}")


if __name__ == "__main__":
    asyncio.run(main())