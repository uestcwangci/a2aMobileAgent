import uvicorn
from fastapi import FastAPI, BackgroundTasks, status
from fastapi.responses import JSONResponse

from data_class import AgentRequest
from dingtalk import DingTalkHelper
from my_utils.logger_util import logger

app = FastAPI()

udid_list = ['localhost:5560', 'localhost:5561', 'localhost:5562', 'localhost:5563', 'localhost:5564',
             'localhost:5565', 'localhost:5566', 'localhost:5567', 'localhost:5568', 'localhost:5569']

async def process_agent(agent_req: AgentRequest):
    """异步处理每个 agent"""
    try:
        # 初始化钉钉助手并添加 UV
        helper = DingTalkHelper(udid="localhost:5560")
        result = helper.add_agent_uv(agent_req)
        logger.info(f"Agent 处理结果: {result}")
    except Exception as e:
        logger.error(f"处理 agent 时出错: {str(e)}")

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