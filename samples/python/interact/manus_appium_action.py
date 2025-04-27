import json
import mimetypes
import os
import time
from http.client import HTTPConnection
from typing import Literal, Dict
from urllib.parse import urlparse

from appium.options.android import UiAutomator2Options
from selenium.webdriver.support.wait import WebDriverWait
import requests
from interact.appium_base_action import AppiumBaseAction, AppiumDriverWrapper, session_timeout_seconds
from interact.molecular import Molecular
from my_utils.logger_util import logger

ACCESS_TOKEN = "lGqMusyvAMqNJEJLmgZanGPAgPNdEtNBwZJAnAxndkE"  # 替换为你的DingTalk token

_open_app_dict = {
    "dingtalk": {"app_package": "com.alibaba.android.rimet", "app_activity": ".biz.LaunchHomeActivity"}
}

def upload_file_source_to_cdn(source: str | bytes, file_type: Literal['image', 'video'], filename: str = None) -> str:
    if isinstance(source, str):
        if not source or not os.path.exists(source):
            raise ValueError('Invalid file path')
        file_name = os.path.basename(source)
        mime_type = mimetypes.guess_type(source)[0] or ('image/png' if file_type == 'image' else 'video/mp4')
        with open(source, 'rb') as f:
            file_content = f.read()
    else:  # 处理二进制数据
        if not isinstance(source, bytes):
            raise ValueError('Source must be a file path or bytes')
        file_name = filename or f"upload_{int(time.time())}.{'png' if file_type == 'image' else 'mp4'}"
        mime_type = 'image/png' if file_type == 'image' else 'video/mp4'
        file_content = source

    if file_type not in ['image', 'video']:
        raise ValueError('Invalid file type. Must be "image" or "video"')
    if file_type == 'video':
        os.makedirs(os.path.join('logs', 'recordVideos'), exist_ok=True)

    try:
        boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
        headers = {
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Authorization': f'Bearer {ACCESS_TOKEN}'
        }
        body = []
        body.append(f'--{boundary}')
        body.append(f'Content-Disposition: form-data; name="file"; filename="{file_name}"')
        body.append(f'Content-Type: {mime_type}')
        body.append('')
        body_bytes = '\r\n'.join(body).encode() + b'\r\n' + file_content + f'\r\n--{boundary}--\r\n'.encode()

        url = urlparse('https://devtool.dingtalk.com/vscode/uploadFile')
        conn = HTTPConnection(url.netloc)
        conn.request('POST', url.path, body=body_bytes, headers=headers)
        response = conn.getresponse()
        if response.status != 200:
            raise ValueError(f'Upload failed with status code: {response.status}')
        response_data = json.loads(response.read().decode())
        if 'cdnUrl' not in response_data:
            raise ValueError('Upload response missing CDN URL')
        print(f"{'image' if file_type == 'image' else 'video'} upload success {response_data['cdnUrl']}")
        return response_data['cdnUrl']
    except Exception as error:
        print('upload fail:', str(error))
        raise


def on_timeout(device_id):
    logger.warning(f"{device_id} Timeout or session disconnect detected！")

"""
在同一个设备（相同的 udid）上，即使 session ID 不同，它们的 newCommandTimeout 也会相互影响。这是因为：
不同设备（不同 udid）的 session 是相互独立的，它们的 newCommandTimeout 不会互相影响
1.实际上这些 session 共享同一个底层的 UiAutomator2 服务器实例（因为是同一个设备）
2.最新创建的 session 会重置计时器，但计时器是在设备级别共享的
3.当任何一个 session 的计时器触发超时，可能会影响到该设备上的其他 session
"""
class AppiumSessionManager:
    def __init__(self):
        self.base_url = 'http://localhost:4723'

    def cleanup_device_sessions(self, udid):
        try:
            sessions = requests.get(f'{self.base_url}/sessions').json()['value']
            for session in sessions:
                if session['capabilities'].get('udid') == udid or \
                        session['capabilities'].get('appium:udid') == udid:
                    requests.delete(f'{self.base_url}/session/{session["id"]}')
                    logger.info(f"Cleaned up session for {udid}: {session['id']}")
        except Exception as e:
            logger.error(f"Error cleaning up sessions for {udid}: {e}")

    def create_session(self, desired_caps):
        udid = desired_caps.get('appium:udid')
        self.cleanup_device_sessions(udid)
        driver = AppiumDriverWrapper(self.base_url,
                                     options=UiAutomator2Options().load_capabilities(desired_caps),
                                     timeout_seconds=session_timeout_seconds,
                                     callback=lambda: on_timeout(udid))
        return driver

session_manager = AppiumSessionManager()

class AppiumAction(AppiumBaseAction):
    def __init__(self, udid):
        super().__init__(udid)
        self.driver = session_manager.create_session(self.desired_caps)
        logger.info("Appium driver initialized")
        time.sleep(3)
        self.molecular = Molecular(self.udid, driver=self.driver)
        self.show_action_pointer()

    def _check_driver_state(self):
        """检查 driver 状态的综合方法"""
        try:
            if not self.driver:
                logger.debug("Driver instance is None")
                return False

            # 检查 session_id
            if not self.driver.session_id:
                logger.debug("No active session ID")
                return False

            # 尝试执行一个简单的命令
            try:
                self.driver.current_activity
                return True
            except Exception as e:
                logger.debug(f"Failed to execute command: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error checking driver state: {str(e)}")
            return False

    def safe_quit(self):
        """安全地关闭 driver"""
        if not self.driver:
            return {'success': True, 'message': 'Driver already None'}

        try:
            self.driver.quit()
            self.driver = None
            self.molecular = None
            return {'success': True, 'message': 'Driver quit successfully'}
        except Exception as e:
                return {
                    'success': False,
                    'message': f'Failed to quit driver {str(e)}'
                }

    def execute(self, action: str, params: Dict = None) -> Dict:
        logger.info(f"Executing action: #{action}# with params: {params}")
        # 检查超时事件
        if self.driver and self.driver.timeout_event.is_set():
            return {"message": "Timeout occurred previously, please start again", "success": False, "timeout": True}
        try:
            if self.driver is None:
                return {"message": "Error: Appium driver not started", "success": False}
            if action == "home":
                # 重新启动应用
                self.home()
                return {"message": "Successfully returned to home page", "success": True}
            elif action == "back":
                # 返回
                self.driver.back()
                return {"message": "Successfully back", "success": True}
            elif action == "screenshot" or action == "screen_shot":
                # 获取 PNG 二进制数据
                start_time = time.time()
                screenshot_png = self.driver.get_screenshot_as_png()
                logger.info("Screenshot captured cost=%.2fs", time.time() - start_time)
                try:
                    # 直接上传二进制数据
                    cdn_url = upload_file_source_to_cdn(screenshot_png, "image", filename="screenshot.png")
                    return {"message": f"Screenshot captured url:{cdn_url}", "screenshot": cdn_url, "success": True}
                except Exception as e:
                    logger.error(f"Error uploading screenshot: {str(e)}")
                    return {"message": f"Error uploading screenshot: {str(e)}", "success": False}
            elif action == "wait":
                seconds = params.get("value", 2)
                if seconds > 0:
                    time.sleep(seconds)
                    return {"message": f"Waited for {seconds} seconds", "success": True}
                return {"message": "Error: Invalid wait time", "success": False}
            elif action == "click":
                x = params.get("x")
                y = params.get("y")
                if x is None or y is None:
                    return {"message": "Error: Missing x or y coordinates", "success": False}
                self.click(x, y)
                return {"message": f"Clicked at ({x}, {y})", "success": True}
            elif action == "double_click":
                x = params.get("x")
                y = params.get("y")
                if x is None or y is None:
                    return {"message": "Error: Missing x or y coordinates", "success": False}
                self.double_click(x, y)
                return {"message": f"Double clicked at ({x}, {y})", "success": True}
            elif action == "long_press":
                x = params.get("x")
                y = params.get("y")
                if x is None or y is None:
                    return {"message": "Error: Missing x or y coordinates", "success": False}
                self.long_press(x, y)
                return {"message": f"Long pressed at ({x}, {y})", "success": True}
            elif action == "type":
                x = params.get("x")
                y = params.get("y")
                text = params.get("value")
                if text is None:
                    return {"message": "Error: type Missing value", "success": False}
                self.type(x, y, text)
                return {"message": "Text input successful", "success": True}
            elif action == "scroll":
                from_coords = params.get("start", [])
                to_coords = params.get("end", [])
                if not from_coords or not to_coords:
                    return {"message": "Error: Missing from or to coordinates", "success": False}
                self.scroll(from_coords, to_coords)
                return {"message": "Scroll successful", "success": True}
            elif action == "startScreenStreaming":
                stream_args = {
                    "host": "0.0.0.0",
                    "quality": 45,
                    "bitRate": 500000,
                    "considersRotation": True
                }
                self.driver.execute_script("mobile: startScreenStreaming", stream_args)
                return {"message": f"Screen streaming started at http://121.43.49.135:8093/", "success": True}
            elif action == "stopScreenStreaming":
                self.driver.execute_script("mobile: stopScreenStreaming")
                return {"message": "Screen streaming stopped", "success": True}
            elif action == "open_app":
                app_name = (params.get("app_name") or "").lower()
                app_package = _open_app_dict[app_name]["app_package"]
                app_activity = _open_app_dict[app_name]["app_activity"]
                self.open_app(app_package, app_activity)
                return {"message": f"Open app {app_name} success", "success": True}
            elif action == "dingtalk_open":
                return {"message": "dingtalk_open", "success": True}
            elif action == "press_enter" or action == "press_search":
                self.driver.press_keycode(66)  # 66 is the keycode for the KEYCODE_ENTER/DONE/SEARCH
                return {"message": "Enter key pressed", "success": True}
            else:
                result = self.molecular.execute(action, params)
                molecular_msg = result["message"]
                success = result["success"]
                return {"message": molecular_msg, "success": success}
        except Exception as e:
            if self.driver and self.driver.timeout_event and self.driver.timeout_event.is_set():
                self.driver = None
                self.molecular = None
                return {"message": f"Timeout occurred please start again: {str(e)}", "success": False, "timeout": True}
            # Replace the selected line with this
            logger.error(f"Execution error: {str(e)}")
            return {"message": f"Error: {str(e)}", "success": False}

    def _show_toast(self, message):
        if self.molecular:
            self.molecular.show_toast(message)

    def _quit(self):
        self.dismiss_action_pointer()
        self.driver.terminate_app(self.desired_caps["appium:appPackage"])
        self.safe_quit()
        logger.info("Appium driver quit")