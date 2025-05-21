# server.py
import asyncio
from typing import Dict
from aliyun.instance_manager import InstanceManager
from mcp.server.fastmcp import FastMCP
import asyncio

# Create an MCP server
mcp = FastMCP("MobileActionMCP")

instance_manager = InstanceManager()

# Add an addition tool
@mcp.tool()
async def execute_action(instance_id: str, action: str, params: Dict) -> Dict:
    """perform actions on phone"""
    async def _execute_action():
        appium_action = instance_manager.get_or_create_client(instance_id)
        result = appium_action.execute(action, params)
        if isinstance(result, dict) and result.get("timeout"):
            # 超时导致的实例被删除，重新创建实例
            appium_action = instance_manager.get_or_create_client(instance_id)
            result = appium_action.execute(action, params)
        return result

    task = asyncio.create_task(_execute_action())
    return await task


@mcp.tool()
async def get_screenshot(instance_id: str) -> str:
    """get a screenshot of the phone(720w*1280h)"""
    async def _get_screenshot():
        appium_action = instance_manager.get_or_create_client(instance_id)
        result = appium_action.execute("screenshot")
        if isinstance(result, dict) and result.get("timeout"):
            # 超时导致的实例被删除，重新创建实例
            appium_action = instance_manager.get_or_create_client(instance_id)
            result = appium_action.execute("screenshot")
        if isinstance(result, dict) and result.get("screenshot"):
            return result["screenshot"]
        return None

    task = asyncio.create_task(_get_screenshot())
    return await task

if __name__ == "__main__":
    # Start the server
    mcp.settings.port=8001
    mcp.run(transport="sse")