import time
from typing import Dict
from urllib.parse import urlparse

import requests
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy

from data_class import AgentRequest
from interact.appium_base_action import AppiumBaseAction, AppiumDriverWrapper, session_timeout_seconds
from interact.molecular import Molecular
from my_utils.logger_util import logger


def on_timeout(device_id):
    logger.warning(f"{device_id} Timeout or session disconnect detected！")

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

class DingTalkHelper(AppiumBaseAction):
    def __init__(self, udid: str):
        super().__init__(udid=udid)
        self.desired_caps['appPackage'] = 'com.alibaba.android.rimet'
        self.desired_caps['appActivity'] = '.biz.LaunchHomeActivity'
        self.driver = session_manager.create_session(self.desired_caps)
        self.molecular = Molecular(self.udid, driver=self.driver)
        logger.info("Appium driver initialized")

    def add_agent_uv(self, req: AgentRequest) -> Dict[str, str]:
        try:
            if not req:
                return {'success': False, 'message': 'req is empty'}
            process_name = []
            for agent_item in req.agent:
                parsed_url = urlparse(agent_item.url)
                if not parsed_url.scheme or not parsed_url.netloc:
                    return {'success': False, 'message': f'Invalid URL format: {agent_item.url}'}
                self.call_jsapi("biz.util", "openLink", {"url": parsed_url})
                time.sleep(10)  # 等待页面加载
                process_name.append(agent_item.name)
                self.driver.back()  # 返回上一页
                time.sleep(2)
            self.molecular.execute("enter_group_chat", {"value":"五常街道第一届弱智杯羽毛球赛"})
            self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/layout_text_content", timeout=10).click()
            self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/rich_edit_text", timeout=10).send_keys(
                f"📈Agent UV涨涨涨: {'，'.join(process_name)}")
            self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/btn_send", timeout=10).click()
            return {'success': True, 'message': 'Agent UV added successfully'}
        except:
            return {'success': False, 'message': 'Invalid URL format'}
        finally:
            self.safe_quit()


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