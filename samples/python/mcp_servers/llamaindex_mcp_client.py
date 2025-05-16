import asyncio
from typing import Optional

from dotenv import load_dotenv
from llama_index.tools.mcp import BasicMCPClient
from llama_index.tools.mcp import (
    aget_tools_from_mcp_url,
)
from llama_index.tools.mcp.base import McpToolSpec

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.client: Optional[BasicMCPClient] = None
        self.tools: Optional[dict[str, McpToolSpec]] = None

    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server

        Args:
        """
        self.client = BasicMCPClient(server_url)
        self.tools = await aget_tools_from_mcp_url(
            server_url,
            client=self.client,
        )
        for tool in self.tools:
            print(tool.metadata.name)


async def main():

    client = MCPClient()
    try:
        await client.connect_to_server("http://118.178.191.176:8001/sse")
    except Exception as e:
        print(f"Error connecting to server: {e}")


if __name__ == "__main__":
    asyncio.run(main())