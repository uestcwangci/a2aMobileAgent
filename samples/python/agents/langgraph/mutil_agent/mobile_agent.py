import json
import time
from typing import Any, AsyncIterable, Callable
from typing import Dict, List
from typing import Literal

from langchain.schema import HumanMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from common.types import AgentSkill
from my_utils.logger_util import logger
from aliyun.instance_manager import InstanceManager

memory = MemorySaver()
instance_manager = InstanceManager()

mobile_skill = AgentSkill(
            id="execute_action",
            name="Mobile Device Interaction Tool",
            description="""
根据用户任务和当前手机屏幕截图，选择合适的工具和参数，完成操作。

execute_action 工具支持以下操作：click, type, long_press, screenshot, double_click, back, scroll, open_app, press_enter, press_search, done。
每次调用 execute_action 后，你会收到结果，包含操作状态和新的截图 URL（例如 {"success": True, "message": "Clicked at (100, 100)", "screenshot": "https://xx.png"}）。
手机截图大小是720(height)*1280(width)，屏幕左上角是(0,0)，你需要返回合理的坐标

你的任务是：
1. 分析截图中的 UI 元素（按钮、文本框、图标等）。
2. 结合用户任务（例如“给 Bob 发短信”），选择合适的操作和参数。
3. 调用 execute_action，处理返回结果，基于新截图继续推理。
4. 如果任务完成，调用 execute_action("done", {})。
5. 如果截图无效或任务无法继续，说明原因并请求用户输入。
        """,
            tags=[
                "mobile automation",
                "touch interaction",
                "app navigation",
                "text input",
                "screen capture",
                "gesture control"
            ],
            examples=[
                "给小明发消息，说你好",
                "打开小明的聊天窗口",
                "帮我总结容器与开放群消息"
            ]
        )

@tool
def execute_action(action: str, params: Dict, runnable_config: RunnableConfig) -> Dict:
    """
   在移动设备上执行指定操作。

:param action: 要执行的操作，可用操作包括：
    - click：单次点击指定坐标
    - type：在指定坐标输入文本
    - long_press：在指定坐标长按
    - screenshot：如果屏幕截图无法识别，可以尝试重新截取屏幕
    - double_click：在指定坐标双击
    - back：返回上一界面
    - scroll：从起始坐标滑动到结束坐标，或按指定方向滑动
    - open_app：打开或切换到指定应用
    - press_enter：发送回车键
    - press_search：发送搜索键
    - done：完成任务

:param params: 操作所需的参数，根据操作类型不同：
    - click/double_click/long_press：{"x": int, "y": int}
    - type：{"x": int, "y": int, "text": str}
    - screenshot/back/press_enter/press_search/done：{}
    - scroll：{"start": [int, int], "end": [int, int]} 或 {"direction": str}（如 "up"）
    - open_app：{"app_name": str}

:param runnable_config: 运行时上下文配置

:return: 包含执行结果的字典：
    - 成功时：{"success": True, "message": "操作信息", "screenshot": image_url}
    - 失败时：{"success": False, "message": "错误信息", "screenshot": image_url}
    其中screenshot是每次操作后，手机产生相应变化后的截图

示例：
    execute_action("click", {"x": 100, "y": 100})
    >>> {"success": True, "message": "Clicked at (100, 100)", "screenshot": "https://opencdn.dingtalk.net/6ca_screenshot.png"}
"""
    ## meta_data = {"instanceId": instanceId}
    configurable = runnable_config['configurable']
    metadata = configurable["metadata"]
    if not metadata:
        return {
            "success": False,
            "message": "Invalid session ID or session not found."
        }

    instance_id = metadata.get("instanceId")
    logger.info(f"instance_id: {instance_id}")
    appium_action = instance_manager.get_or_create_client(instance_id)
    result = appium_action.execute(action, params)
    # 延迟2s后
    time.sleep(2)
    result["screenshot"] = appium_action.execute("screenshot")["screenshot"]
    return result


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class MobileInteractionAgent:
    SUPPORTED_CONTENT_TYPES: List[str] = ["text", "text/plain"]

    SYSTEM_INSTRUCTION = """
    你是一个 MobileInteractionAgent，负责以拟人化的方式操作手机，使用点击、输入、滑动等工具与应用程序交互。你的目标是根据用户任务，选择合适的工具和参数，完成手机操作。

    ## 工作流程
    1. **分析**：理解用户任务，将其分解为可执行步骤。
    2. **行动**：根据屏幕截图，选择合适的工具和参数执行下一步。
    3. **执行**：使用工具执行选定的动作。
    4. **评估**：检查动作结果，确认成功或识别错误。
    5. **循环**：重复分析-行动-执行-评估，直到任务完成。
    6. **结束**：返回最终结果。

    ## 规则
    - 手机截图大小是720(height)*1280(width)
    - 始终从打开相关应用开始。如果任务未指定应用，默认打开钉钉(DingTalk)。
    - 钉钉支持发送消息、创建日程、访问工作台等操作。
    - 必须模拟人类行为，使用点击、输入、滑动等工具操作应用。
    - 打开应用后，无需重复打开，除非任务明确要求。
    - 记录已执行的步骤，方便后续决策。
    - 如果用户询问与手机操作无关的问题，礼貌回应：“抱歉，我只能协助处理手机操作相关任务。”
    - 不得将工具用于手机操作以外的目的。
    - 根据情况设置响应状态：
      - `input_required`：需要用户提供更多信息。
      - `error`：处理请求时发生错误。
      - `completed`：任务已完成。
    - 当前时间：{time}

    ## 输出
    - 如果上一步成功，思考下一步应使用哪个工具及参数。
    - 如果上一步失败，分析错误并提出纠正措施。
    - 提供已执行步骤的简要总结和当前任务状态。
    """

    def __init__(self):
        self.model = ChatOpenAI(model="claude35_sonnet2")
        self.tools = [execute_action]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat
        )

    def invoke(self, query: str, image_url:str, session_id: str, metadata: Dict=None) -> Dict[str, Any]:
        logger.info(f"invoke: {session_id} {query}")
        config = {"configurable": {"thread_id": session_id}}
        inputs = {
            "messages": [
                HumanMessage(
                    content=[
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                )
            ]
        }
        self.graph.invoke(input=inputs, config=RunnableConfig(configurable={"thread_id": session_id, "metadata": metadata}))
        return self.get_agent_response(config)

    async def stream(self, query: str, image_url:str, session_id: str, metadata: Dict=None) -> AsyncIterable[Dict[str, Any]]:
        logger.info(f"stream {session_id} {query}")
        config = {"configurable": {"thread_id": session_id}}
        metadata = metadata or {}

        # 初始消息历史
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {"type": "image", "source_type": "url", "url": image_url}
                ]
            )
        ]

        while True:
            # 流式处理当前消息历史
            async for item in self.graph.astream(
                    {"messages": messages},
                    config=RunnableConfig(configurable={"thread_id": session_id, "metadata": metadata}),
                    stream_mode="values"
            ):
                message = item["messages"][-1]
                logger.info(f"Message: {message}")

                if isinstance(message, AIMessage) and message.tool_calls:
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Processing action..."
                    }
                elif isinstance(message, ToolMessage):
                    # 解析 ToolMessage 的 content，提取 screenshot URL
                    try:
                        tool_result = json.loads(message.content)
                        new_image_url = tool_result.get("screenshot")
                        if not new_image_url:
                            logger.warning("No screenshot URLcom URL found in ToolMessage, requesting new screenshot")
                            # 触发 screenshot 操作
                            messages.append(
                                AIMessage(content="",
                                    tool_calls=[{"name": "execute_action", "args": {"action": "screenshot", "params": {}}}]))
                            yield {
                                "is_task_complete": False,
                                "require_user_input": False,
                                "content": "Requesting new screenshot..."
                            }
                            continue
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse ToolMessage content: {message.content}")
                        yield {
                            "is_task_complete": False,
                            "require_user_input": True,
                            "content": "Invalid tool response format. Please try again."
                        }
                        break

                    # 清理消息历史，保留任务和最新截图
                    messages = [
                        msg for msg in messages
                        if isinstance(msg, HumanMessage) and msg.content[0]["text"] == query
                    ]
                    # 追加最近的工具调用和结果（可选：限制历史长度）
                    recent_messages = [msg for msg in item["messages"] if isinstance(msg, (AIMessage, ToolMessage))][-2:]
                    messages.extend(recent_messages)
                    # 追加新截图
                    messages.append(
                        HumanMessage(
                            content=[
                                {"type": "text", "text": f"New screenshot received for task: {query}"},
                                {"type": "image", "source_type": "url", "url": new_image_url}
                            ]
                        )
                    )

                    logger.info(f"Updated screenshot URL: {new_image_url}...")
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Processed tool response, new screenshot: {new_image_url}"
                    }

            # 检查任务是否完成
            response = self.get_agent_response(config)
            yield response
            if response.get("is_task_complete", False) or response.get("require_user_input", False):
                break

    def get_agent_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"get_agent_response: {config}")
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again."
        }