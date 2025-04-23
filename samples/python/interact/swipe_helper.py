import time
from enum import Enum
from typing import List

from appium.webdriver.common.appiumby import AppiumBy

from my_utils.logger_util import logger


class ScrollDirection(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


def _is_point_in_bounds(point: List[int], bounds: str) -> bool:
    """检查点是否在元素范围内"""
    # bounds format: "[left,top][right,bottom]"
    try:
        bounds = bounds.replace('][', ',').replace('[', '').replace(']', '')
        left, top, right, bottom = map(int, bounds.split(','))
        return (left <= point[0] <= right) and (top <= point[1] <= bottom)
    except:
        return False


class ScrollAction:
    def __init__(self, driver):
        self.driver = driver

    def find_scrollable_element_by_point(self, point: List[int]) -> str | None:
        """通过坐标找到可滚动元素的resource-id"""
        try:
            scrollable_elements = []
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='android.widget.ScrollView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='android.widget.GridView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='androidx.viewpager.widget.ViewPager'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='android.widget.ListView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='android.widget.RecyclerView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='androidx.recyclerview.widget.RecyclerView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='android.widget.HorizontalScrollView'))
            scrollable_elements.append(
                self.driver.find_elements(by=AppiumBy.CLASS_NAME, value='androidx.core.widget.NestedScrollView'))
            scrollable_elements = [elem for sublist in scrollable_elements for elem in sublist]  # 扁平化列表

            for element in scrollable_elements:
                bounds = element.get_attribute('bounds')
                if _is_point_in_bounds(point, bounds):
                    return element.get_attribute('resource-id')
            return None
        except Exception as e:
            logger.warning(f"Error finding scrollable element: {str(e)}")
        return None

    def swipe(self, direction: str, point: List[int]) -> bool:
        """执行滚动操作"""
        try:
            direction = ScrollDirection(direction.upper())
        except ValueError:
            logger.warning(f"Invalid direction: {direction}")
            return False

        resource_id = self.find_scrollable_element_by_point(point)
        if not resource_id:
            logger.warning("No scrollable element found at the given point")
            return False

        # 构建滚动命令
        base_command = f'new UiScrollable(new UiSelector().resourceId("{resource_id}"))'

        # 根据方向设置滚动参数
        if direction in [ScrollDirection.UP, ScrollDirection.DOWN]:
            command = f'{base_command}.setAsVerticalList()'
            command += '.scrollForward()' if direction == ScrollDirection.UP else '.scrollBackward()'
        else:
            command = f'{base_command}.setAsHorizontalList()'
            command += '.scrollForward()' if direction == ScrollDirection.LEFT else '.scrollBackward()'

        try:
            from appium.webdriver.common.appiumby import AppiumBy
            self.driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR, command)
            time.sleep(0.5)  # 等待滚动完成
            return True
        except Exception as e:
            logger.warning(f"Scroll failed: {str(e)}")
            return False
