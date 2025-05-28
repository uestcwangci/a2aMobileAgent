import asyncio
import time
from typing import Dict
from urllib.parse import urlparse

import aiohttp
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

    async def cleanup_device_sessions(self, udid):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f'{self.base_url}/sessions') as response:
                    sessions = (await response.json())['value']
                    for session_data in sessions:
                        if session_data['capabilities'].get('udid') == udid or \
                                session_data['capabilities'].get('appium:udid') == udid:
                            async with session.delete(f'{self.base_url}/session/{session_data["id"]}') as del_response:
                                logger.info(f"Cleaned up session for {udid}: {session_data['id']}")
            except Exception as e:
                logger.error(f"Error cleaning up sessions for {udid}: {e}")

    async def create_session(self, desired_caps):
        udid = desired_caps.get('appium:udid')
        await self.cleanup_device_sessions(udid)
        loop = asyncio.get_running_loop()
        # 将 Appium 会话创建放入线程池，因为 Appium 客户端可能是阻塞的
        driver = await loop.run_in_executor(
            None,
            lambda: AppiumDriverWrapper(
                self.base_url,
                options=UiAutomator2Options().load_capabilities(desired_caps),
                timeout_seconds=session_timeout_seconds,
                callback=lambda: on_timeout(udid)
            )
        )
        return driver


session_manager = AppiumSessionManager()


class DingTalkHelper(AppiumBaseAction):
    def __init__(self, udid: str):
        super().__init__(udid=udid)
        self.desired_caps['appPackage'] = 'com.alibaba.android.rimet'
        self.driver = None  # 延迟初始化
        self.molecular = None
        logger.info("DingTalkHelper initialized")

    async def initialize_driver(self):
        """异步初始化 Appium driver"""
        if self.driver is None:
            self.driver = await session_manager.create_session(self.desired_caps)
            self.molecular = Molecular(self.udid, driver=self.driver)
            logger.info("Appium driver initialized")

    async def add_agent_uv(self, req: AgentRequest) -> Dict[str, str]:
        try:
            await self.initialize_driver()  # 确保 driver 已初始化
            if not req:
                return {'success': False, 'message': 'req is empty'}

            # 异步等待 jsapi 广播 service 加载完成
            await asyncio.sleep(10)
            process_name = []
            for i in range(len(req.url)):
                if not req.url[i] or not req.name[i]:
                    continue
                try:
                    parsed_url = urlparse(req.url[i])
                except:
                    continue
                if not parsed_url.scheme or not parsed_url.netloc:
                    continue

                # 假设 call_jsapi 是同步的，放入线程池
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.call_jsapi("biz.util", "openLink", {"url": parsed_url.geturl()})
                )
                await asyncio.sleep(6)  # 等待页面加载

                # 假设 tap 是同步的，放入线程池
                await loop.run_in_executor(None, lambda: self.driver.tap([(627, 459)]))
                await asyncio.sleep(2)
                await loop.run_in_executor(None, lambda: self.driver.tap([(620, 686)]))
                await asyncio.sleep(2)
                process_name.append(req.name[i])
                await loop.run_in_executor(None, self.driver.back)
                await asyncio.sleep(2)

            # 假设 execute 和 wait_for_find 是同步的，放入线程池
            await loop.run_in_executor(
                None,
                lambda: self.molecular.execute("enter_group_chat", {"value": "容器与开放快乐拼"})
            )
            element = await loop.run_in_executor(
                None,
                lambda: self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/layout_text_content", timeout=10)
            )
            await loop.run_in_executor(None, lambda: element.click())
            element = await loop.run_in_executor(
                None,
                lambda: self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/rich_edit_text", timeout=10)
            )
            await loop.run_in_executor(
                None,
                lambda: element.send_keys(f"Agent UV: {'，'.join(process_name)}")
            )
            element = await loop.run_in_executor(
                None,
                lambda: self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/btn_send", timeout=10)
            )
            await loop.run_in_executor(None, lambda: element.click())
            return {'success': True, 'message': 'Agent UV added successfully'}
        except Exception as e:
            logger.error(f"Error in add_agent_uv: {str(e)}")
            return {'success': False, 'message': f'Invalid URL format or error: {str(e)}'}
        finally:
            await self.safe_quit()

    async def safe_quit(self):
        """异步安全关闭 driver"""
        if not self.driver:
            return {'success': True, 'message': 'Driver already None'}

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.driver.terminate_app, self.desired_caps["appium:appPackage"])
            await loop.run_in_executor(None, self.driver.quit)
            self.driver = None
            self.molecular = None
            return {'success': True, 'message': 'Driver quit successfully'}
        except Exception as e:
            return {'success': False, 'message': f'Failed to quit driver: {str(e)}'}