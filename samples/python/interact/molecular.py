import time

from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait

from my_utils.logger_util import logger
from interact.appium_base_action import AppiumBaseAction
from interact.swipe_helper import ScrollAction


class Molecular(AppiumBaseAction):
    def __init__(self, udid, driver=None):
        super().__init__(udid)
        self.driver = driver

    def _enter_chat(self, chat_type: str, value: str):
        """
        进入聊天（单聊或群聊）
        chat_type: "contact（联系人）" || "group（群组） || "workapp（工作台）"
        value: 搜索的名称
        """
        if value is None:
            raise ValueError(f"Value for enter_{chat_type} is required")

        # 回到首页
        self.call_jsapi("internal.automator", "navigateToHome")
        time.sleep(2)

        # 点击搜索按钮
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/search_btn").click()

        # 根据类型选择不同的标签
        tab_text = "工作台" if chat_type == "workapp" else "联系人" if chat_type == "contact" else "群组"
        tab = self.scroll_into_text("com.alibaba.android.rimet:id/lv_tabs",
                                    tab_text, direction="horizontal", timeout=10)
        tab.click()

        # 输入搜索内容
        self.wait_for_find(by=AppiumBy.ID, value="android:id/search_src_text", timeout=15).send_keys(value)

        if tab_text == "工作台":
            # Web页面
            # 打印当前状态
            print("Available Contexts:", self.driver.contexts)

            # 等待页面加载
            time.sleep(2)
            # 等待 WebView 可用
            WebDriverWait(self.driver, 30).until(lambda d: len(d.contexts) > 1)
            # 点击第一个搜索结果
            self.wait_for_find(
                by=AppiumBy.CSS_SELECTOR,
                value='.dtm-list-item:first-of-type .dtm-button-button',
                timeout=5
            ).click()
            return {"message": f"Entered {chat_type} chat with {value}", "success": True}
        else:
            # Native页面
            # 点击第一个搜索结果
            self.wait_for_find(
                by=AppiumBy.ANDROID_UIAUTOMATOR,
                value='new UiSelector().resourceId("com.alibaba.android.rimet:id/list_view").childSelector(new UiSelector().index(1))',
                timeout=15
            ).click()
            return {"message": f"Entered {chat_type} chat with {value}", "success": True}

    def _reset_group_mute(self, mute: bool = False):
        self.scroll_into_text("com.alibaba.android.rimet:id/scroll_view", "消息免打扰")
        parent = self.wait_for_find(AppiumBy.ANDROID_UIAUTOMATOR, "new UiSelector().description(\"消息免打扰\")")
        toggle = parent.find_element(AppiumBy.ID, "com.alibaba.android.rimet:id/menu_item_toggle")
        if mute != (toggle.get_attribute('checked') == 'true'):
            toggle.click()
        return {"message": "Reset group mute", "success": True}

    def _reset_work_status(self):
        # 点击头像
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/my_avatar").click()
        # 点击工作状态
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/user_person_status").click()
        # 滑动到最顶部
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        # 点击“无状态”
        self.wait_for_find(AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().text("无状态")').click()
        # 点击确定
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/more_text").click()
        return {"message": "Reset work status", "success": True}

    def _set_work_status_to_vacation(self):
        # 点击头像
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/my_avatar").click()
        # 点击工作状态
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/user_person_status").click()
        # 滑动到最顶部
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        self.scroll((0, 250), (0, 1000))
        # 点击“无状态”
        self.wait_for_find(AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().textContains("休假")').click()
        # 点击确定
        self.wait_for_find(AppiumBy.ID, "com.alibaba.android.rimet:id/more_text").click()
        return {"message": "Set work status to vacation", "success": True}

    def execute(self, action, data):
        value = data.get("value")

        if action == "enter_single_chat":
            return self._enter_chat("contact", value)
        elif action == "enter_group_chat":
            return self._enter_chat("group", value)
        elif action == "enter_app":
            return self._enter_chat("workapp", value)
        elif action == "show_toast":
            return self.show_toast(value)
        elif action == "show_action_pointer":
            return self.show_action_pointer()
        elif action == "dismiss_action_pointer":
            return self.dismiss_action_pointer()
        elif action == "move_action_pointer":
            return self.move_action_pointer(data.get("x"), data.get("y"))
        elif action == "reset_group_mute":
            # 重置群聊消息免打扰
            return self._reset_group_mute(value)
        elif action == "reset_work_status":
            # 重置工作状态为无状态
            return self._reset_work_status()
        elif action == "set_work_status_to_vacation":
            # 设置工作状态为休假
            return self._set_work_status_to_vacation()
        elif action == "swipe":
            scroll_action = ScrollAction(self.driver)
            # 示例调用
            point = data.get("point")
            direction = data.get("direction")
            result = scroll_action.swipe(direction, point)
            return {"message": "Text swipe action performed", "success": result}
        elif action == "open_link":
            # 打开链接
            url = data.get("url")
            self.call_jsapi(service_name="biz.util", action_name="openLink", params={"url": url})
            return {"message": f"Opened link: {url}", "success": True}
        else:
            logger.error(f"Unknown molecular: {action}")
            return {"message": f"Error: Unsupported actionType '{action}'", "success": False}

    def show_toast(self, message: str):
        if not self.driver:
            logger.error("Driver is None, cannot show toast")
            return {"message": "Driver is None, cannot show toast", "success": False}
        try:
            self.call_static(class_name="com.alibaba.android.dingtalkbase.tools.AndTools", method="showToast",
                             params=[{"type": "string", "value": message}])
            return {"message": f"Showed toast: {message}", "success": True}
        except Exception as e:
            logger.error(f"Failed to show toast: {e}")
            return {"message": f"Failed to show toast: {e}", "success": False}
