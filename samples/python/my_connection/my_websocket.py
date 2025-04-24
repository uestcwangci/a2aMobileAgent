import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import websocket

from my_utils.logger_util import logger
from interact.manus_appium_action import AppiumAction

executor = ThreadPoolExecutor(max_workers=5)

class WebSocketClient:
    def __init__(self, url, max_retries=None, initial_retry_delay=5, udid=None):
        self.url = url
        self.ws = None
        self.is_running = True
        self.heartbeat_thread = None
        self.max_retries = max_retries  # None表示无限重试
        self.retry_delay = initial_retry_delay  # 初始重试延迟（秒）
        self.retry_count = 0
        self.reconnect_event = threading.Event()
        self.appium_action = None
        self.udid = udid

    def send_heartbeat(self):
        """发送心跳包"""
        while self.ws and self.is_running and not self.reconnect_event.is_set():
            try:
                self.ws.send('{"type": "heartbeat"}')
                time.sleep(30)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                self.reconnect_event.set()  # 通知主线程需要重连
                break

    def on_open(self, ws):
        """连接建立时的回调"""
        logger.info("Connected to external WebSocket server")
        self.retry_count = 0  # 连接成功，重置重试计数
        self.retry_delay = 5  # 重置重试延迟
        self.reconnect_event.clear()
        self.appium_action = AppiumAction(self.udid)

        # 启动心跳线程
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            # 如果之前的线程还在运行，先确保它结束
            self.heartbeat_thread.join(timeout=1)

        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()
        ws.send(json.dumps({"status": "ready"}))

    def on_message(self, ws, message):
        """收到消息时的回调"""
        try:
            logger.info(f"Received from server: {message}")
            response = self.process_message(message)
            if response:
                ws.send(response)
        except Exception as e:
            logger.error(f"Message processing error: {e}")

    def on_error(self, ws, error):
        """发生错误时的回调"""
        logger.error(f"WebSocket client error: {str(error)}")
        self.reconnect_event.set()  # 标记需要重连

    def on_close(self, ws, close_status_code, close_msg):
        """连接关闭时的回调"""
        logger.info(f"WebSocket client closed: {close_status_code} - {close_msg}")
        self.reconnect_event.set()  # 标记需要重连

    def cleanup(self):
        """清理资源"""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
            self.ws = None
        if self.appium_action:
            self.appium_action.quit()
            self.appium_action = None

    def run(self):
        """运行WebSocket客户端"""
        while self.is_running:
            try:
                # 先检查重试次数
                if self.max_retries and self.retry_count >= self.max_retries:
                    logger.info(f"Reached the maximum number of retries {self.max_retries}, stopped reconnecting")
                    break

                logger.info(f"Try connecting to {self.url}")
                self.reconnect_event.clear()
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close
                )

                # 运行WebSocket客户端
                self.ws.run_forever(ping_interval=0)  # 禁用ping/pong机制，使用自定义心跳

                # 如果连接断开，检查是否应该重新连接
                if not self.is_running:
                    break

                # 增加重试计数
                self.retry_count += 1

                # 清理旧连接资源
                self.cleanup()

                # 如果达到最大重试次数，直接退出
                if self.max_retries and self.retry_count >= self.max_retries:
                    logger.info(f"Reached the maximum number of retries {self.max_retries}, stopped reconnecting")
                    break

                # 只有在未达到最大重试次数时才等待重连
                retry_time = min(self.retry_delay * (2 ** min(self.retry_count, 5)), 300)
                logger.info(f"Disconnected, trying to reconnect {self.retry_count} times after {retry_time} seconds...")
                time.sleep(retry_time)

            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
                self.cleanup()

                # 增加重试计数
                self.retry_count += 1

                # 如果达到最大重试次数，直接退出
                if self.max_retries and self.retry_count >= self.max_retries:
                    logger.info(f"Reached the maximum number of retries {self.max_retries}, stopped reconnecting")
                    break

                # 只有在未达到最大重试次数时才等待重连
                retry_time = min(self.retry_delay * (2 ** min(self.retry_count, 5)), 300)
                time.sleep(retry_time)

    def stop(self):
        """停止客户端"""
        self.is_running = False
        self.reconnect_event.set()  # 唤醒任何等待的线程
        self.cleanup()
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=2)

    def process_message(self, message):
        """处理消息的核心逻辑，线程安全"""
        def build_response(action_type, action_uuid, ext, action=None, message="Action executed", data=None):
            """构建标准响应格式"""
            response = {
                "action": action_type,
                "actionUuid": action_uuid,
                "ext": ext,
                "data": data or {"execAction": action} if action else {},
                "message": message
            }
            return json.dumps(response)

        # 解析JSON
        try:
            action_data = json.loads(message)
            logger.debug(f"Parsed action data: {action_data}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return build_response("error", None, None, message="Invalid JSON format")

        # 提取基本字段
        action = action_data.get("action")
        action_uuid = action_data.get("actionUuid")
        ext = action_data.get("ext")

        # 验证必要字段
        if not action:
            logger.error("No action specified in message")
            return build_response("execFail", action_uuid, ext, action, "No action specified in message")

        if self.appium_action is None:
            logger.error("Appium driver not started")
            return build_response("execFail", action_uuid, ext, action, "Appium driver not started")

        result = self.appium_action.execute(action_data)
        return self._process_action_result(result, action_data)

    def _process_action_result(self, result, action_data):
        """处理动作执行结果"""
        action_uuid = action_data.get("actionUuid")
        ext = action_data.get("ext")
        action = action_data.get("action")

        def build_response(action_type, message="Action executed", data=None):
            response = {
                "action": action_type,
                "actionUuid": action_uuid,
                "ext": ext,
                "data": data or {"execAction": action, "url": result.get("screenshot", "")},
                "message": message
            }
            return json.dumps(response)

        logger.info(f"Action result: {result}")
        return build_response(
            "execSuccess" if result.get("success") else "execFail",
            result.get("message", "Action executed"),
            {"execAction": action, "url": result.get("screenshot", "")}
        )