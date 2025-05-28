import uvicorn
from fastapi import FastAPI, BackgroundTasks, status
from fastapi.responses import JSONResponse
import asyncio
from typing import List, Dict
from data_class import AgentRequest
from dingtalk import DingTalkHelper
from my_utils.logger_util import logger
import time
import random  # 新增 random 模块用于随机延迟

app = FastAPI()

udid_list = ['localhost:5560', 'localhost:5561', 'localhost:5562', 'localhost:5563', 'localhost:5564',
             'localhost:5565', 'localhost:5566', 'localhost:5567', 'localhost:5568', 'localhost:5569',
             'localhost:5570', 'localhost:5571', 'localhost:5572', 'localhost:5573', 'localhost:5574']

async def process_single_agent(agent_req: AgentRequest, udid: str) -> Dict[str, str]:
    """处理单个设备的 agent"""
    try:
        start_time = time.time()
        logger.info(f"设备 {udid} 开始处理 at {start_time}")
        helper = DingTalkHelper(udid=udid)
        result = await helper.add_agent_uv(agent_req)  # Ensure this is awaited if async
        end_time = time.time()
        logger.info(f"设备 {udid} 处理完成 in {end_time - start_time:.2f}s: {result}")
        return result
    except Exception as e:
        error_msg = f"设备 {udid} 处理 agent 时出错: {str(e)}"
        logger.error(error_msg)
        return {"success": False, "message": error_msg}

async def process_agent(agent_req: AgentRequest):
    semaphore = asyncio.Semaphore(5)  # 限制为 5 个并发任务
    async def limited_process_single_agent(agent_req: AgentRequest, udid: str, delay: float) -> Dict[str, str]:
        # 在任务开始前添加延迟
        await asyncio.sleep(delay)
        async with semaphore:
            return await process_single_agent(agent_req, udid)

    # 为每个任务生成一个随机延迟（例如 0 到 120 秒之间），CPU削峰
    tasks = [limited_process_single_agent(agent_req, udid, random.uniform(0, 120)) for udid in udid_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 记录所有结果
    for udid, result in zip(udid_list, results):
        if isinstance(result, Exception):
            logger.error(f"设备 {udid} 执行失败: {str(result)}")
        else:
            logger.info(f"设备 {udid} 执行完成: {result}")

@app.post("/agent")
async def handle_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """处理 agent POST 请求"""
    background_tasks.add_task(process_agent, request)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "success", "message": "请求已接收，正在后台处理"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)