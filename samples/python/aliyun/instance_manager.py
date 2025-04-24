import asyncio
import threading
from typing import Any

from aliyun.client import AliyunClient
from my_utils.logger_util import logger
from interact.manus_appium_action import AppiumAction

"""
Singleton decorator to ensure only one instance of InstanceManager exists.
"""
def singleton(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

DEFAULT_ADB_PORT = 5555

@singleton
class InstanceManager:
    def __init__(self):
        self.aliyun_client = AliyunClient()
        self.all_clients: dict[str, str] = {} # instance_id -> network_interface_ip
        self.active_clients: dict[str, AppiumAction] = {}
        self.active_clients_lock = threading.Lock()
        self.session_to_instance = {}
        asyncio.run(self.load_modules())

    async def load_modules(self):
        instances = await self.aliyun_client.describe_android_instances()
        for instance in instances:
            self.all_clients[instance.android_instance_id] = f"{instance.network_interface_ip}:{DEFAULT_ADB_PORT}"
        logger.info(f"all clients: {self.all_clients}")

    def get_interface_ip(self, instance_id: str) -> str:
        return self.all_clients.get(instance_id, "")

    def get_all_instances(self) -> dict[Any, Any]:
        return self.all_clients

    def get_available_instances(self) -> set[str]:
        with self.active_clients_lock:
            return self.all_clients.keys() - self.active_clients.keys()

    def get_or_create_client(self, instance_id: str) -> AppiumAction:
        with self.active_clients_lock:
            self.active_clients[instance_id] = self.active_clients.get(instance_id) or AppiumAction(udid=self.get_interface_ip(instance_id))
            return self.active_clients[instance_id]

    def release_client(self, instance_id: str):
        with self.active_clients_lock:
            appium_client = self.active_clients.pop(instance_id, None)
            if appium_client:
                appium_client.safe_quit()
                logger.info(f"Released client for instance {instance_id}")

    def clear(self):
        with self.active_clients_lock:
            for appium_client in self.active_clients.values():
                appium_client.safe_quit()
            self.active_clients.clear()
            self.session_to_instance.clear()