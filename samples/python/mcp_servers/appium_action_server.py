# server.py
from typing import Dict
from aliyun.instance_manager import InstanceManager
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("MobileActionMCP")

instance_manager = InstanceManager()

# Add an addition tool
@mcp.tool()
async def execute_action(instance_id: str, action: str, params: Dict) -> Dict:
    """perform actions on phone"""
    appium_action = instance_manager.get_or_create_client(instance_id)
    result = appium_action.execute(action, params)
    return result

@mcp.tool()
async def get_screenshot(instance_id: str) -> str:
    """get a screenshot of the phone(720w*1280h)"""
    appium_action = instance_manager.get_or_create_client(instance_id)
    return appium_action.execute("screenshot")["screenshot"]

if __name__ == "__main__":
    # Start the server
    mcp.settings.port=8001
    mcp.run(transport="sse")