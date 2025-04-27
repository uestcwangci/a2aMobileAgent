from common.server import A2AServer
from common.types import AgentCard, AgentCapabilities, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth
from agents.langgraph.task_manager import AgentTaskManager
import click
import os
import logging
from dotenv import load_dotenv

from agents.langgraph.mutil_agent.mobile_agent import MobileInteractionAgent, mobile_skill

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option("--host", "host", default="0.0.0.0")
@click.option("--port", "port", default=10000)
def main(host, port):
    """Starts the Currency Agent server."""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise MissingAPIKeyError("OPENAI_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)

        agent_card = AgentCard(
            name="Mobile Automation Assistant",
            description="""An intelligent agent specialized in Android device automation.

        Core Features:
        - Precise touch interaction execution
        - Text input and navigation
        - Screenshot capture and analysis
        - Complex task sequencing
        - Error handling and recovery
        - Natural interaction patterns

        Designed for automated mobile testing, UI navigation, and task automation while maintaining human-like interaction patterns.""",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=MobileInteractionAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=MobileInteractionAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[mobile_skill]
        )

        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=MobileInteractionAgent(), notification_sender_auth=notification_sender_auth),
            host=host,
            port=port,
        )

        server.app.add_route(
            "/.well-known/jwks.json", notification_sender_auth.handle_jwks_endpoint, methods=["GET"]
        )

        logger.info(f"Starting server on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)


if __name__ == "__main__":
    main()
