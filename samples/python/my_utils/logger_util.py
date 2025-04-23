import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    自定义的TimedRotatingFileHandler类，用于直接创建指定格式的日志文件
    """
    def __init__(self, log_dir, base_name, *args, **kwargs):
        self.log_dir = log_dir
        self.base_name = base_name
        # 生成当前日期的文件名
        current_file = self._get_current_filename()
        super().__init__(current_file, *args, **kwargs)

    def _get_current_filename(self):
        """生成当前日期的文件名"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.log_dir, f"{current_date}-{self.base_name}")

    def doRollover(self):
        """
        重写文件滚动方法，在滚动时使用新的日期创建文件
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # 使用新的日期创建新文件
        self.baseFilename = self._get_current_filename()
        if not self.delay:
            self.stream = self._open()

def setup_logger(log_dir='logs'):
    """
    配置日志系统，按天分割日志文件。

    Args:
        log_dir (str): 日志存储目录，默认为 'logs'

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器（按天分割）
    file_handler = CustomTimedRotatingFileHandler(
        log_dir=log_dir,
        base_name='trace.log',
        when='midnight',  # 每天午夜分割
        interval=1,  # 间隔1天
        backupCount=30,  # 保留30天的日志
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 获取根记录器并配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空默认处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logging.getLogger(__name__)

logger = setup_logger()