import json
from typing import AsyncIterable, Dict, Any

from langchain.tools import Tool
from langchain.tools import tool

from agents.langgraph.mutil_agent.base_agent import BaseAgent
from aliyun.instance_manager import InstanceManager
from session_manager import SessionManager
from common.types import AgentSkill

instance_manager = InstanceManager()
session_manager = SessionManager()

mobile_skill = AgentSkill(
            id="execute_action",
            name="Mobile Device Interaction Tool",
            description="""
A comprehensive tool for executing precise mobile device actions.

Capabilities:
- Execute touch interactions (click, double_click, long_press)
- Input text at specific coordinates
- Capture and analyze screenshots
- Navigate apps and screens
- Perform scrolling operations
- Simulate system keys (enter, search)

This tool supports natural, human-like interaction patterns while maintaining precise control over mobile device operations.
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
def execute_action(action: str, params: json, **kwargs) -> json:
    """
   Execute a specific action on the mobile device.

   :param action: The action to be performed. Available actions include:
       - click: Single tap at specified coordinates
       - type: Input text at specified coordinates
       - long_press: Press and hold at coordinates
       - screen_shot: Capture current screen and return image URL
       - double_click: Two quick taps at coordinates
       - back: Return to previous screen
       - scroll: Scroll from start to end coordinates
       - open_app: Launch or switch to an app
       - press_enter: Send keycode for enter
       - press_search: Send keycode for search
       - done: Complete the task

   :param params: Parameters for the action, varying by action type:
       - For click/double_click/long_press: {"x": int, "y": int}
       - For type: {"x": int, "y": int, "text": str}
       - For scroll: {"start": [x, y], "end": [x, y]}
       - For open_app: {"app_name": str}
       - For screen_shot/back/press_enter/press_search/done: {}

    :param context: Contextual information, including instance ID and thread ID.

   :return: Dictionary containing execution result:
       - On success: {"success": True, "message": "action info", "screenshot":${img_url} # Only for screen_shot action }
       - On failure: {"success": False, "message": "error message"}

   Example:
       execute_action("click", {"x": 100, "y": 100})
       {"success": True, "message": "Clicked at (100, 100)"}

       execute_action("type", {"x": 100, "y": 100, "text": "Hello"})
       {"success": True, "message": "Typed 'Hello' at (100, 100)"}

       execute_action("scroll", {"start": [100, 200], "end": [100, 500]})
       {"success": True, "message": "Scrolled from [100, 200] to [100, 500]"}
           """
    config = kwargs.get("config", {})
    ## config = {"configurable": {"thread_id": session_id}}
    session_id = config.get("configurable", {}).get("thread_id", "unknown")
    print("Thread ID from config: {session_id}")

    ## meta_data = {"instanceId": instanceId}
    meta_data = session_manager.get_meta(session_id)
    if not meta_data:
        return {
            "success": False,
            "message": "Invalid session ID or session not found."
        }
    instance_id = meta_data.get("instanceId")
    appium_action = instance_manager.get_or_create_client(instance_id)
    return appium_action.execute(action, params)

class MobileInteractionAgent(BaseAgent):
    def __init__(self) -> None:
        tools = [execute_action]  # 创建工具列表
        super().__init__(tools=tools)  # 使用关键字参数传递


    @property
    def SYSTEM_INSTRUCTION(self) -> str:
        return """
You are a specialized mobile automation assistant designed to interact with Android devices through Appium. You have access to precise device control through the execute_action tool.

Core Capabilities:
1. Execute mobile actions using the 'execute_action' tool
2. Analyze screen content through screenshots
3. Perform precise touch interactions
4. Navigate apps and input text
5. Execute sequential operations

Available Actions:
- click: Precise tapping at coordinates
- type: Text input at specific locations
- long_press: Extended press gestures
- screen_shot: Capture screen state
- double_click: Quick double taps
- back: Navigate to previous screen
- scroll: Smooth scrolling between points
- open_app: Application launching
- press_enter: Enter key simulation
- press_search: Search key simulation
- done: Task completion

Operation Guidelines:
1. Start complex tasks by capturing the current screen state using screen_shot
2. Verify the success of each action by checking the response
3. Use precise coordinates for interactions
4. Break down complex tasks into sequential steps
5. Handle errors gracefully by checking success status
6. End task sequences with the 'done' action

Error Handling:
- If an action fails, analyze the error message and adjust strategy
- If screen recognition is unclear, request a new screenshot
- If coordinates are uncertain, provide detailed reasoning

Remember to:
1. Always check action results through the returned success/message
2. Use screenshots for visual confirmation when needed
3. Maintain context across multiple actions
4. Follow human-like interaction patterns
5. Stay within the defined action set
6. If no application is specified, open_app(params: DingTalk) is used by default.

IMPORTANT: Only use the actions and parameters exactly as defined in the execute_action tool documentation. If asked to perform actions outside these capabilities, explain that you are limited to the available mobile interactions.

For all actions, provide clear reasoning for your choices and explain your strategy when executing multi-step tasks."""


# 自定义工具
# class ExecuteActionTool(Tool):
#     name = "execute_action"
#     description = """
#    Execute a specific action on the mobile device.
#
#    :param action: The action to be performed. Available actions include:
#        - click: Single tap at specified coordinates
#        - type: Input text at specified coordinates
#        - long_press: Press and hold at coordinates
#        - screen_shot: Capture current screen and return image URL
#        - double_click: Two quick taps at coordinates
#        - back: Return to previous screen
#        - scroll: Scroll from start to end coordinates
#        - open_app: Launch or switch to an app
#        - press_enter: Send keycode for enter
#        - press_search: Send keycode for search
#        - done: Complete the task
#
#    :param params: Parameters for the action, varying by action type:
#        - For click/double_click/long_press: {"x": int, "y": int}
#        - For type: {"x": int, "y": int, "text": str}
#        - For scroll: {"start": [x, y], "end": [x, y]}
#        - For open_app: {"app_name": str}
#        - For screen_shot/back/press_enter/press_search/done: {}
#
#     :param context: Contextual information, including instance ID and thread ID.
#
#    :return: Dictionary containing execution result:
#        - On success: {"success": True, "message": "action info", "screenshot":${img_url} # Only for screen_shot action }
#        - On failure: {"success": False, "message": "error message"}
#
#    Example:
#        execute_action("click", {"x": 100, "y": 100})
#        {"success": True, "message": "Clicked at (100, 100)"}
#
#        execute_action("type", {"x": 100, "y": 100, "text": "Hello"})
#        {"success": True, "message": "Typed 'Hello' at (100, 100)"}
#
#        execute_action("scroll", {"start": [100, 200], "end": [100, 500]})
#        {"success": True, "message": "Scrolled from [100, 200] to [100, 500]"}
#            """
#     args_schema = MobileInteractionAgent
#
#     def _run(self, action: str, params: dict, **kwargs) -> dict:
#         # 在这里可以访问 config（通过 kwargs）
#         config = kwargs.get("config", {})
#         ## config = {"configurable": {"thread_id": session_id}}
#         session_id = config.get("configurable", {}).get("thread_id", "unknown")
#         print("Thread ID from config: {session_id}")
#
#         ## meta_data = {"instanceId": instanceId}
#         meta_data = session_manager.get_meta(session_id)
#         if not meta_data:
#             return {
#                 "success": False,
#                 "message": "Invalid session ID or session not found."
#             }
#         instance_id = meta_data.get("instanceId")
#         appium_action = instance_manager.get_or_create_client(instance_id)
#         return appium_action.execute(action, params)