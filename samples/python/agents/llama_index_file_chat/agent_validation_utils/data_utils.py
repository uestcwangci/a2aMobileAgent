import json
import os

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock
from llama_index.llms.openai_like import OpenAILike

from agents.llama_index_file_chat.agent_validation_utils.prompt_constants import VALIDATION_PROMPT, FINAL_RES_VALIDATION_PROMPT
from agents.llama_index_file_chat.agent_validation_utils.task_completion_analysis import TaskCompletionAnalyzer
import requests

from my_utils.logger_util import logger

from pydantic import BaseModel, Field, model_validator

class CheckResultDecision(BaseModel):
    reasoning: str = Field(description="判断是否符合预期的理由")
    is_complete: bool = Field(description="是否符合预期")

class ActionStepChecker:
    task_completion_analyzer = TaskCompletionAnalyzer()

    def __init__(self, task:str, task_id: str):
        self.task = task
        self.task_id = task_id
        # self.llm = OpenAILike(
        #     model="gpt-4o-0806",
        #     api_base=os.getenv("AI_STUDIO_BASE"),
        #     api_key=os.getenv("AI_STUDIO_API_KEY"),
        #     context_window=200000,
        #     is_chat_model=True,
        #     is_function_calling_model=True,
        # ).as_structured_llm(CheckResultDecision)

        self.llm = OpenAILike(
            model="qwen2.5-vl-72b-instruct",
            api_base=os.getenv("AI_STUDIO_BASE"),
            api_key=os.getenv("AI_STUDIO_API_KEY"),
            context_window=131072,
            max_tokens=129024,
            is_chat_model=True,
            is_function_calling_model=False,
        ).as_structured_llm(CheckResultDecision)

        self.process_screenshot_list = []

    async def check(self, action: str, action_params:dict, before_action_screenshot_url: str, after_action_screenshot_url: str):
        before_action_screenshot_path = self._url_to_path(before_action_screenshot_url)
        after_action_screenshot_path = self._url_to_path(after_action_screenshot_url)

        #action_params 转换为json字符串
        action_params_str = json.dumps(action_params)
        # 使用 LLM 检查是否符合预期
        messages = [
            ChatMessage(role="system", content=VALIDATION_PROMPT),
            ChatMessage(role="user", blocks=[
                ImageBlock(path=before_action_screenshot_path),
                ImageBlock(path=after_action_screenshot_path),
                
                TextBlock(text=f"分析动作执行前后的截图，判断是否符合预期：{action} 参数：{action_params_str}")
            ])
        ]

        # 调用 LLM
        response = await self.llm.achat(messages)

        check_result : CheckResultDecision = response.raw

        logger.info(f"kkk TASK:{self.task} TASK_ID:{self.task_id} ActionStepChecker check: 是否符合预期： {check_result.is_complete} 理由： {check_result.reasoning}")
        
    async def check_final_result(self, diff_jo: dict, history: list):
        '''
        检查最终的结果是否满足预期
        需要信息：
        1. 历史步骤的截图
        2. 预期结果的描述
        '''
        # 将process_screenshot_list转换为ImageBlock的数组
        content_blocks = [ImageBlock(path=screenshot_path) for screenshot_path in self.process_screenshot_list]

        # process_screenshot_list 打印出来
        logger.info(f"kkk screenshot_list TASK:{self.task} TASK_ID:{self.task_id} process_screenshot_list: {self.process_screenshot_list}")
        content_blocks.append(TextBlock(text="\n".join(diff_jo["rubrics"])))

        messages = [
            ChatMessage(role="system", content=FINAL_RES_VALIDATION_PROMPT),
            ChatMessage(role="user", blocks=content_blocks)
        ]

        # 调用 LLM
        response = await self.llm.achat(messages)

        check_result : CheckResultDecision = response.raw

        logger.info(f"kkk TASK:{self.task} TASK_ID:{self.task_id} check_final_result check: 是否符合预期： {check_result.is_complete} 理由： {check_result.reasoning}")

        ActionStepChecker.task_completion_analyzer.add_task_result(self.task, self.task_id, check_result.is_complete, check_result.reasoning)

    def save_process_screenshot(self, screenshot_path: str):
        self.process_screenshot_list.append(screenshot_path)

    @classmethod
    def gen_stats_report(self):
        ActionStepChecker.task_completion_analyzer.generate_pie_chart()
        ActionStepChecker.task_completion_analyzer.save_stats_to_csv()
        ActionStepChecker.task_completion_analyzer.save_task_details_to_csv()

    def _url_to_path(self, url: str) -> str:
        temp_dir = os.path.join(os.getcwd(), "tmp", "screenshots")
        # 如果文件夹不存在就创建
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_image_path = os.path.join(temp_dir, f"{hash(url)}.png")

        img_response = requests.get(url, verify=False)

        with open(temp_image_path, 'wb') as f:
            f.write(img_response.content)

        return temp_image_path