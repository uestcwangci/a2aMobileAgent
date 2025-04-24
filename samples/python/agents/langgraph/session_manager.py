import asyncio
import json
import threading
from typing import Any, Dict

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

@singleton
class SessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._session_to_meta: Dict[str, json] = {}

    def put_meta(self, session_id: str, meta: json):
        with self._lock:
            self._session_to_meta[session_id] = meta

    def get_meta(self, session_id: str) -> json:
        with self._lock:
            return self._session_to_meta.get(session_id)

    def clear(self):
        with self._lock:
            self._session_to_meta.clear()