from abc import ABC, abstractmethod
from typing import Any, Dict, AsyncIterable, List, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel
from typing import Literal

memory = MemorySaver()

# 回调处理器
class ConfigCallbackHandler(BaseCallbackHandler):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def on_tool_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        # 将 config 注入到工具的输入中
        inputs["config"] = self.config


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

class BaseAgent(ABC):
    SUPPORTED_CONTENT_TYPES: List[str] = ["text", "text/plain"]

    @property
    @abstractmethod
    def SYSTEM_INSTRUCTION(self) -> str:
        pass

    def __init__(self, tools: List[Callable] = None):
        """
        初始化 BaseAgent

        Args:
            tools: 必须是被 @tool 装饰器修饰的函数列表
        """
        self.model = ChatOpenAI(model="claude35_sonnet2")
        self.tools = tools if tools is not None else []

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat
        )

    def invoke(self, query: str, session_id: str) -> Dict[str, Any]:
        config = {"configurable": {"thread_id": session_id}}
        bound_graph = self.graph.bind(config=config)
        bound_graph.invoke({"messages": [("user", query)]}, config=config, callbacks=ConfigCallbackHandler(config))
        return self.get_agent_response(config)

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": session_id}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if isinstance(message, AIMessage) and message.tool_calls:
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing tool response...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
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
            "content": "We are unable to process your request at the moment. Please try again.",
        }