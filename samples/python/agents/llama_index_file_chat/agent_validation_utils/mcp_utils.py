import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
from my_utils.logger_util import logger
import json


INSTANCE_ID = 'acp-9nfh0qgrttkkfi6kc'

'''
工具列表：
    [Tool(name = 'execute_action', description = 'perform actions on phone', inputSchema = {
        'properties': {
            'instance_id': {
                'title': 'Instance Id',
                'type': 'string'
            },
            'action': {
                'title': 'Action',
                'type': 'string'
            },
            'params': {
                'additionalProperties': True,
                'title': 'Params',
                'type': 'object'
            }
        },
        'required': ['instance_id', 'action', 'params'],
        'title': 'execute_actionArguments',
        'type': 'object'
    }, annotations = None), 
    Tool(name = 'get_screenshot', description = 'get a screenshot of the phone(720w*1280h)', inputSchema = {
        'properties': {
            'instance_id': {
                'title': 'Instance Id',
                'type': 'string'
            }
        },
        'required': ['instance_id'],
        'title': 'get_screenshotArguments',
        'type': 'object'
    }, annotations = None)]

'''
class MCPClient:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def execute_action(self, action: str, params: dict):
        async with self._lock:
            async with sse_client('http://118.178.191.176:8001/sse') as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    res = await session.call_tool('execute_action', {
                        'instance_id': INSTANCE_ID,
                        'action': action,
                        'params': params
                    })
                    jo = json.loads(res.content[0].text)
                    
                    logger.info(f"kkk execute_action: {jo}")
                    return jo

    async def get_screenshot(self):
        async with self._lock:
            async with sse_client('http://118.178.191.176:8001/sse') as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    res = await session.call_tool('get_screenshot', {
                        'instance_id': INSTANCE_ID
                    })
                    logger.info(f"kkk get_screenshot: {res}")
                    return res.content[0].text

    @classmethod
    async def create(cls):
        return cls()

