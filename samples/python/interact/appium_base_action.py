import json
import time
from threading import Event, Thread, Timer
from typing import Union, Dict, List

from appium.options.common import AppiumOptions
from appium.webdriver import Remote
from appium.webdriver import WebElement
from appium.webdriver.common.appiumby import AppiumBy
from selenium.common import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from appium.webdriver.applicationstate import ApplicationState

from my_utils.logger_util import logger

session_timeout_seconds = 1800 # 30min无操作，自动释放appium session

class AppiumDriverWrapper(Remote):
    def __init__(self, command_executor, options: AppiumOptions=None, timeout_seconds=session_timeout_seconds,
                 callback=None):
        # 调用父类 Remote 的初始化方法
        super().__init__(command_executor=command_executor, options=options)
        # 添加自定义属性
        logger.info('__init__ AppiumDriverWrapper')
        self.timeout_seconds = timeout_seconds
        self.last_command_time = time.time()
        self.callback = callback if callback else lambda: print("Timeout callback executed")
        self.timeout_event = Event()
        self._initializing = False  # 初始化完成后关闭标志

    def __getattribute__(self, name):
        # 直接从 self 获取属性或方法（会通过 Python 的属性查找机制访问父类）
        parent_getattribute = super(AppiumDriverWrapper, self).__getattribute__
        attr = parent_getattribute(name)

        try:
            initializing = parent_getattribute('_initializing')
        except AttributeError:
            # 如果 _initializing 不存在，说明对象还未初始化完成
            return attr

        # 如果是方法，包装它以添加超时检查
        internal_method = ['execute', 'terminate_app', 'activate_app', 'quit', 'stop_client']
        if callable(attr) and not name.startswith('__') and not name.startswith('_') and not name in internal_method:
            def wrapper(*args, **kwargs):
                logger.debug(f"__wrapper__ called for method: {name}")
                timeout_seconds = parent_getattribute('timeout_seconds')
                last_command_time = parent_getattribute('last_command_time')
                callback = parent_getattribute('callback')
                timeout_event = parent_getattribute('timeout_event')

                current_time = time.time()
                elapsed_time = current_time - last_command_time
                if elapsed_time > timeout_seconds:
                    if callback:
                        callback()
                    timeout_event.set()
                    raise Exception(f"Timeout {timeout_seconds} seconds exceeded for method: {name}")

                result = attr(*args, **kwargs)
                parent_getattribute('__dict__')['last_command_time'] = time.time()
                return result

            return wrapper
        return attr


def get_element_center(element):
    rect = element.rect
    center_x = rect['x'] + (rect['width'] / 2)
    center_y = rect['y'] + (rect['height'] / 2)
    return center_x, center_y

class AppiumBaseAction:
    def __init__(self, udid = None):
        self.driver = None
        self.molecular = None
        self.udid = udid
        self.desired_caps = {
            "platformName": "Android",
            "appium:automationName": "Uiautomator2",
            "appium:unicodeKeyboard": True, # 使用 Unicode 输入法
            "appium:resetKeyboard": True, # 测试结束后重置输入法
            "appium:noReset": True,  # 防止重置应用
            "appium:forceAppLaunch": True,  # 每次启动强制重启app
            "appium:newCommandTimeout": session_timeout_seconds,  # 5分钟
            "appium:autoGrantPermissions": True, # 自动授予权限
            "appium:udid": udid
        }
        self.webview_context = "WEBVIEW_com.alibaba.android.rimet"  # 可配置的 WebView 上下文

    def _switch_context_for_by(self, by: str) -> None:
        """
        根据定位策略自动切换上下文。
        """
        # WebView 支持的定位方式
        webview_strategies = [AppiumBy.CSS_SELECTOR]

        current_context = self.driver.current_context
        available_contexts = self.driver.contexts

        if by in webview_strategies:
            if self.webview_context in available_contexts:
                if current_context != self.webview_context:
                    logger.info(f"Switching context from '{current_context} to '{self.webview_context}")
                    self.driver.switch_to.context(self.webview_context)
            else:
                raise Exception(f"WebView context '{self.webview_context}' not available. Available contexts: {available_contexts}")
        else:
            if current_context != "NATIVE_APP":
                logger.info(f"Switching context from '{current_context} to 'NATIVE_APP'")
                self.driver.switch_to.context("NATIVE_APP")

    def show_action_pointer(self):
        if not self.driver:
            logger.error("Driver is None, cannot show action pointer")
            return {"message": "Driver is None, cannot show action pointer", "success": False}
        try:
            self.call_jsapi(service_name="internal.automator", action_name="showActionPointer")
            return {"message": "Showed action pointer", "success": True}
        except Exception as e:
            logger.error(f"Failed to show action pointer: {e}")
            return {"message": f"Failed to show action pointer: {e}", "success": False}

    def dismiss_action_pointer(self):
        if not self.driver:
            logger.error("Driver is None, cannot dismiss action pointer")
            return {"message": "Driver is None, cannot dismiss action pointer", "success": False}
        try:
            self.call_jsapi(service_name="internal.automator", action_name="dismissActionPointer")
            return {"message": "Dismiss action pointer", "success": True}
        except Exception as e:
            logger.error(f"Failed to dismiss action pointer: {e}")
            return {"message": f"Failed to dismiss action pointer: {e}", "success": False}

    def move_action_pointer(self, x: int, y: int):
        if not self.driver:
            logger.error("Driver is None, cannot move action pointer")
            return {"message": "Driver is None, cannot move action pointer", "success": False}
        try:
            self.call_jsapi(service_name="internal.automator", action_name="moveActionPointer", params={"x": x, "y": y})
            return {"message": "Move action pointer", "success": True}
        except Exception as e:
            logger.error(f"Failed to move action pointer: {e}")
            return {"message": f"Failed to move action pointer: {e}", "success": False}

    def wait_for_find(self, by: str = AppiumBy.ID, value: Union[str, Dict, None] = None, timeout: int = 5) -> WebElement:
        """
        等待并查找单个元素，自动切换上下文。
        :param by: 定位策略 (e.g., AppiumBy.ID, AppiumBy.CSS_SELECTOR)
        :param value: 定位值
        :param timeout: 等待超时时间（单位：秒）
        :return: 找到的元素
        """
        self._switch_context_for_by(by)
        element = WebDriverWait(self.driver, timeout).until(
            lambda x: self.driver.find_element(by=by, value=value)
        )
        if element:
            center = get_element_center(element)
            move_thread = Thread(target=self.move_action_pointer, args=(center[0], center[1]))
            move_thread.start()
        return element

    def wait_for_finds(self, by: str = AppiumBy.ID, value: Union[str, Dict, None] = None, timeout: int = 5) -> List[WebElement]:
        """
        等待并查找多个元素，自动切换上下文。
        :param by: 定位策略 (e.g., AppiumBy.ID, AppiumBy.CSS_SELECTOR)
        :param value: 定位值
        :param timeout: 等待超时时间（单位：秒）
        :return: 找到的元素列表
        """
        self._switch_context_for_by(by)
        elements = WebDriverWait(self.driver, timeout).until(
            lambda x: self.driver.find_elements(by=by, value=value)
        )
        return elements

    def scroll_into_text(self, parent_value, text, direction="vertical", timeout=10):
        """
        在指定容器内滚动，直到找到包含指定文本的元素并返回。

        Args:
            parent_value: 滚动容器的 resourceId (如 "com.example:id/scroll_view")。
            text: 目标元素的文本内容。
            direction: 滚动方向，"vertical"（垂直，默认）或 "horizontal"（水平）。
            timeout: 查找超时时间（秒），默认 5。

        Returns:
            WebElement: 找到的目标元素。

        Raises:
            NoSuchElementException: 如果滚动容器未找到。
            TimeoutException: 如果目标元素未找到。
            ValueError: 如果方向参数无效。
        """
        # 验证方向参数并构造 UIAutomator 滚动表达式
        scroll_expression = (
            f'new UiScrollable(new UiSelector().resourceId("{parent_value}").scrollable(true))'
        )
        if direction == "horizontal":
            scroll_expression += f'.setAsHorizontalList().scrollTextIntoView("{text}")'
        elif direction == "vertical":
            scroll_expression += f'.scrollTextIntoView("{text}")'
        else:
            logger.error(f"Invalid scroll direction: {direction}")
            raise ValueError(f"Direction must be 'vertical' or 'horizontal', got: {direction}")

        # 查找并返回目标元素
        try:
            target_element = self.wait_for_find(AppiumBy.ANDROID_UIAUTOMATOR, scroll_expression, timeout=timeout)
            logger.info(f"Found target element with text: '{text}' in {direction} scroll")
            return target_element
        except TimeoutException:
            logger.error(f"Could not find element with text '{text}' in {timeout} seconds")
            raise TimeoutException(f"Failed to scroll to element with text '{text}'")

    def _call_native(self, call_type: str, **kwargs):
        """
        Consolidated method to call native functionality via broadcast intents.

        Args:
            call_type: Type of call - "static", "instance", or "jsapi"
            **kwargs: Arguments specific to each call type:
                - static: class_name, method, params (optional)
                - instance: class_name, method, instance_method (optional), params (optional)
                - jsapi: service_name, action_name, params (optional)

        Returns:
            None
        """
        method_call = {"type": call_type}

        if call_type == "static" or call_type == "instance":
            """
            数据结构示例:
            {
                "type": "static",
                "className": "com.alibaba.android.dingtalkbase.tools.AndTools",
                "method": "showToast",
                "params": [
                    {"type": "string", "value": "Hello World"}
                ]
            }
            """
            class_name = kwargs.get("class_name")
            method = kwargs.get("method")

            if not class_name or not method:
                raise ValueError(f"For {call_type} calls, class_name and method are required")

            method_call["className"] = class_name
            method_call["method"] = method
            method_call["params"] = kwargs.get("params")

            if call_type == "instance":
                method_call["instanceMethod"] = kwargs.get("instance_method", "getInstance")

        elif call_type == "jsapi":
            service_name = kwargs.get("service_name")
            action_name = kwargs.get("action_name")

            if not service_name or not action_name:
                raise ValueError("For jsapi calls, service_name and action_name are required")

            method_call["jsapiParams"] = {
                "serviceName": service_name,
                "actionName": action_name,
                "params": kwargs.get("params")
            }
        else:
            raise ValueError(f"Unsupported call type: {call_type}")

        # Convert method_call to a properly escaped JSON string
        json_string = json.dumps(method_call, ensure_ascii=False)
        full_command = f"am broadcast -a appium.to.dingtalk.ACTION -p com.alibaba.android.rimet --receiver-permission com.alibaba.android.rimet.APPIUM_PERMISSION --es methodCall '{json_string}'"
        broadcast_command = {
            "command": full_command
        }

        result = self.driver.execute_script('mobile: shell', broadcast_command)
        logger.info(f"Shell command result: {result}")
        return result

    def call_static(self, class_name: str, method: str, params: List = None):
        """
        调用静态方法。

        Args:
            class_name: 类名。
            method: 方法名。
            params: 参数。

        Returns:
            None
        """
        self._call_native("static", class_name=class_name, method=method, params=params)

    def call_instance(self, class_name: str, method: str, instance_method: str = "getInstance", params: List = None):
        """
        调用实例方法。

        Args:
            class_name: 类名。
            method: 方法名。
            instance_method: 获取实例的方法名。
            params: 参数。

        Returns:
            None
        """
        self._call_native("instance", class_name=class_name, method=method, instance_method=instance_method, params=params)

    def call_jsapi(self, service_name: str, action_name: str, params: Dict = None):
        """
        调用 JSAPI。

        Args:
            service_name: 服务名。
            action_name: 动作名。
            params: 参数。

        Returns:
            None
        """
        self._call_native("jsapi", service_name=service_name, action_name=action_name, params=params)

    def click(self, x: int, y: int) -> None:
        """
        点击指定坐标。
        """
        move_thread = Thread(target=self.move_action_pointer, args=(x, y))
        move_thread.start()
        self.driver.tap([(x, y)], 100)

    def double_click(self, x: int, y: int) -> None:
        """
        双击指定坐标。
        """
        move_thread = Thread(target=self.move_action_pointer, args=(x, y))
        move_thread.start()
        self.driver.tap([(x, y)], 100)
        time.sleep(0.1)
        self.driver.tap([(x, y)], 100)

    def long_press(self, x: int, y: int, duration: int = 1000) -> None:
        """
        长按指定坐标。
        """
        move_thread = Thread(target=self.move_action_pointer, args=(x, y))
        move_thread.start()
        self.driver.tap([(x, y)], duration)

    def type(self, x: int, y: int, text: str) -> None:
        """
        在指定坐标输入文本。
        """
        if x is not None and y is not None:
            self.driver.tap([(x, y)], 100)
            # 动态等待输入框就绪
            try:
                WebDriverWait(self.driver, 2).until(
                    lambda driver: driver.switch_to.active_element.get_attribute("class").endswith("EditText")
                )
                logger.info("Input field is ready")
            except Exception as wait_error:
                logger.warning(f"Failed to wait for input field: {wait_error}")
                time.sleep(1)  # 备用等待

        # 获取当前活跃元素
        element = self.driver.switch_to.active_element
        if element:
            center = get_element_center(element)
            move_thread = Thread(target=self.move_action_pointer, args=(center[0], center[1]))
            move_thread.start()

        # 方法 1：尝试 send_keys
        try:
            element.send_keys(text)
            logger.info("Text input via send_keys succeeded")
        except Exception as e:
            logger.debug(f"Send_keys failed: {e}")
            # 方法 2：使用 ADB 输入（适合英文和简单字符）
            try:
                adb_text = text.replace(" ", "%s")  # 处理空格
                self.driver.execute_script("mobile: shell", {
                    "command": "input",
                    "args": ["text", adb_text]
                })
                logger.info("Text input via ADB succeeded")
            except Exception as adb_error:
                logger.debug(f"ADB input failed: {adb_error}")
                # 方法 3：使用剪贴板输入（支持中文和复杂字符）
                # 先缓存剪贴板的内容
                last_clip_text = self.driver.get_clipboard_text()
                try:
                    self.driver.set_clipboard_text(text)
                    element.click()  # 确保焦点
                    self.driver.execute_script("mobile: shell", {
                        "command": "input",
                        "args": ["keyevent", "279"]  # KEYCODE_PASTE
                    })
                    logger.info("Text input via clipboard succeeded")
                except Exception as clipboard_error:
                    logger.debug(f"Clipboard input failed: {clipboard_error}")
                    # 方法 4：备用方案，使用 JavaScript（H5 页面）
                    try:
                        self.driver.execute_script("arguments[0].value = arguments[1];", element, text)
                        logger.info("Text input via JavaScript succeeded")
                    except Exception as js_error:
                        logger.error(f"JavaScript input failed: {js_error}")
                finally:
                    # 恢复剪贴板内容
                    self.driver.set_clipboard_text(last_clip_text)

    def scroll(self, start: tuple, end: tuple, duration: int = 500) -> None:
        """
        滑动操作。
        """
        move1_thread = Thread(target=self.move_action_pointer, args=(end[0], end[1]))
        move1_thread.start()
        # 延迟执行第二次移动
        delay = duration / 1000 / 2  # 将duration转换为秒并取一半作为延时
        Timer(delay, self.move_action_pointer, args=(start[0], start[1])).start()
        self.driver.swipe(start[0], start[1], end[0], end[1], duration)

    def launch_app(self, app_package: str, app_activity: str) -> None:
        """
        打开指定应用。
        """
        # 综合检查应用状态
        try:
            # 检查应用状态
            app_state = self.driver.query_app_state(app_package)
            if app_state in [ApplicationState.NOT_RUNNING, ApplicationState.NOT_INSTALLED]:
                logger.info(f"App is not running (state: {app_state}), starting it")
                self.driver.activate_app(app_package)
                return

            # 尝试获取当前activity
            try:
                current_activity = self.driver.current_activity
                if not current_activity:
                    logger.info("App might be starting, skipping home action")
                    return
            except Exception:
                logger.info("Unable to get current activity, app might be starting")
                return

            # 执行重启操作
            self.driver.terminate_app(app_package)
            self.driver.activate_app(app_package)

        except Exception as e:
            logger.warning(f"Error during home operation: {str(e)}")
            # 如果出错，尝试直接启动应用
            try:
                self.driver.activate_app(app_package)
            except Exception as e2:
                logger.error(f"Failed to activate app: {str(e2)}")
                raise e

        # 等待主页面加载
        WebDriverWait(self.driver, timeout=30).until(
            lambda driver: driver.current_activity == app_activity)
        time.sleep(3)  # 等待页面完全加载

    def home(self) -> None:
        """
        回到钉钉主页。
        """
        self.call_jsapi("internal.automator", "navigateToHome")