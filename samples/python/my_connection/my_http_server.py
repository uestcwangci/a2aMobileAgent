import asyncio
import threading

from quart import Quart, jsonify, request

from aliyun.instance_manager import InstanceManager
from my_connection.socket_manager import SocketManager
from my_utils.logger_util import logger
from aliyun.client import AliyunClient
from aiocache import Cache
from typing import Dict, Any, List
from my_connection.my_websocket import WebSocketClient
from quart_cors import cors
from urllib.parse import urlparse, parse_qs

http_server = Quart(__name__)
# http_server = cors(http_server, allow_origin="*")
aliyun_client = AliyunClient()
cache = Cache(Cache.MEMORY)

instance_manager = InstanceManager()
socket_manager = SocketManager()

class APIResponse:
    @staticmethod
    def success(data: Any = None, message: str = "success") -> Dict:
        return {
            "code": 200,
            "message": message,
            "data": data
        }

    @staticmethod
    def error(message: str, code: int = 500, data: Any = None) -> Dict:
        return {
            "code": code,
            "message": message,
            "data": data
        }

@http_server.route('/api/getTicket', methods=['GET'])
async def get_ticket():
    try:
        # 从request的query中取instance_id
        instance_id = request.args.get('instanceId')
        if not instance_id:
            response = APIResponse.error(
                message="instanceId is required",
                code=400
            )
            return jsonify(response), 400

        # 尝试从缓存获取
        # cached_data = await cache.get("tickets")
        # if cached_data is not None:
        #     return jsonify(cached_data)

        # 获取新数据
        tickets_model = await aliyun_client.batch_get_acp_connection_ticket()
        if not tickets_model:
            response = APIResponse.error(
                message="No tickets found",
                code=404
            )
            return jsonify(response), 404

        ticket = next((model.ticket for model in tickets_model if model.instance_id == instance_id), None)
        if ticket is None:
            response = APIResponse.error(
                message="Ticket not found",
                code=404
            )
            return jsonify(response), 404
        response = APIResponse.success(data={'ticket': ticket})
        # await cache.set("tickets", response, ttl=60)

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in get_ticket: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=str(e))), 500


@http_server.route('/api/getAvailableInstances', methods=['GET'])
def get_available_devices():
    """获取可用设备列表"""
    try:
        instances = instance_manager.get_available_instances()
        return jsonify(APIResponse.success(data=list(instances)))
    except Exception as e:
        return jsonify(APIResponse.error(message=str(e)))

@http_server.route('/api/getAllInstances', methods=['GET'])
def get_all_devices():
    """获取可用设备列表"""
    try:
        instances = instance_manager.get_all_instances()
        return jsonify(APIResponse.success(data=list(instances)))
    except Exception as e:
        return jsonify(APIResponse.error(message=str(e)))

# 用于管理 WebSocket 客户端连接的字典
websocket_clients = {}
websocket_clients_lock = threading.Lock()

@http_server.route('/android/agent/createAgentConnection', methods=['POST'])
async def create_agent_connection():
    """创建代理连接"""
    try:
        body = await request.get_json()
        ws_address = body.get('wsAddress')

        if not ws_address:
            return jsonify(APIResponse.error(message="wsAddress is required")), 400

        # 解析 URL
        parsed_url = urlparse(ws_address)
        # 解析查询参数
        query_params = parse_qs(parsed_url.query)
        # 获取实例 ID
        instance_id = query_params.get('instanceId', [None])[0]
        if not instance_id:
            return jsonify(APIResponse.error(message="instanceId is required")), 400
        instance_ip = instance_manager.get_all_instances()[instance_id]
        if not instance_ip:
            return jsonify(APIResponse.error(message="instanceId is not valid")), 400

        # 检查是否已经有相同地址的连接
        with websocket_clients_lock:
            if ws_address in websocket_clients:
                # 如果已存在连接，先关闭旧连接
                logger.info(f"Close an existing WebSocket connection: {ws_address}")
                old_client = websocket_clients[ws_address]
                old_client.stop()

        def run_websocket_client():
            """运行WebSocket客户端的线程函数"""
            websocket_client = WebSocketClient(ws_address, max_retries=1, udid=instance_ip)

            # 将客户端添加到管理字典中
            with websocket_clients_lock:
                websocket_clients[ws_address] = websocket_client

            try:
                websocket_client.run()
            finally:
                # 连接结束后从管理字典中移除
                with websocket_clients_lock:
                    if ws_address in websocket_clients:
                        del websocket_clients[ws_address]

        ws_client_thread = threading.Thread(target=run_websocket_client, daemon=True)
        ws_client_thread.start()

        logger.info(f"WebSocket client connection started: {ws_address}")
        return jsonify(APIResponse.success(message=f"WebSocket client started: {ws_address}"))

    except Exception as e:
        logger.error(f"Failed to create WebSocket connection: {e}", exc_info=True)
        return jsonify(APIResponse.error(message=f"Failed to create WebSocket connection: {str(e)}")), 500

@http_server.route('/api/clearCache', methods=['POST'])
async def clear_cache():
    """清除缓存的接口"""
    try:
        await cache.delete("tickets")
        return jsonify(APIResponse.success(message="Cache cleared"))
    except Exception as e:
        return jsonify(APIResponse.error(message=str(e)))


@http_server.route('/live/connectWebsocket', methods=['POST'])
async def live_connect_websocket():
    """清除缓存的接口"""
    try:
        body = await request.get_json()
        ws_address = body.get('wsAddress')
        if not ws_address:
            return jsonify(APIResponse.error(message="wsAddress is required")), 400
        logger.info(f"live/connectWebsocket called with wsAddress: {ws_address}")

        ws_client_thread = threading.Thread(target=socket_manager.force_reconnect, args=(ws_address,), daemon=True)
        ws_client_thread.start()
        return jsonify(APIResponse.success(message=f"WebSocket client started: {ws_address}"))
    except KeyboardInterrupt:
        return jsonify(APIResponse.error(message="WebSocket client interrupted by user"))
    except Exception as e:
        return jsonify(APIResponse.error(message=str(e)))


@http_server.errorhandler(404)
async def not_found(error):
    return jsonify(APIResponse.error(
        message="Resource not found",
        code=404
    )), 404


@http_server.errorhandler(500)
async def internal_error(error):
    return jsonify(APIResponse.error(
        message="Internal server error",
        code=500
    )), 500


# 添加健康检查接口
@http_server.route('/health', methods=['GET'])
async def health_check():
    return jsonify(APIResponse.success(data={"status": "healthy"}))


if __name__ == '__main__':
    http_server.run(host='0.0.0.0', port=5000)