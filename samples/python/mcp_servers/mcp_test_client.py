import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    # methods will go here

    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server

        Args:
        """

        sse_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
        sse, write = sse_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(sse, write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [(tool.name, tool.inputSchema) for tool in tools])

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():

    client = MCPClient()
    try:
        await client.connect_to_server("http://118.178.191.176:8001/sse")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())