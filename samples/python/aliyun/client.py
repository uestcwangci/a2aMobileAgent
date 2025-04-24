# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys

from alibabacloud_eds_aic20230930.models import BatchGetAcpConnectionTicketResponseBodyInstanceConnectionModels, \
    DescribeAndroidInstancesResponseBodyInstanceModel
from dotenv import load_dotenv
from my_utils.logger_util import logger

from typing import List, Optional, Dict

from alibabacloud_eds_aic20230930.client import Client as eds_aic20230930Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_eds_aic20230930 import models as eds_aic_20230930_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_console.client import Client as ConsoleClient
from alibabacloud_tea_util.client import Client as UtilClient


class AliyunClient:
    def __init__(self):
        print("Current working directory:", os.path.dirname(os.path.dirname(__file__)))

        # 确认 .env 文件存在
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        print("Looking for .env file at:", env_path)
        print("File exists:", os.path.exists(env_path))
        load_dotenv()
        try:
            config = open_api_models.Config(
                access_key_id=os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID'],
                access_key_secret=os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
            )
            # Endpoint 配置
            config.endpoint = 'eds-aic.cn-shanghai.aliyuncs.com'
            self.client = eds_aic20230930Client(config)
        except KeyError as e:
            logger.error(f"Missing environment variable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AliyunClient: {e}")
            raise

    async def batch_get_acp_connection_ticket(self) -> Optional[List[BatchGetAcpConnectionTicketResponseBodyInstanceConnectionModels]]:
        """
        获取ACP连接票据
        Returns:
            List[Dict]: 连接模型列表，失败返回None
        """
        try:
            batch_get_acp_connection_ticket_request = eds_aic_20230930_models.BatchGetAcpConnectionTicketRequest(
                end_user_id=os.environ["END_USER_ID"],
                instance_group_id=os.environ['INSTANCE_GROUP_ID']
            )
            runtime = util_models.RuntimeOptions()

            # 发送请求
            resp = await self.client.batch_get_acp_connection_ticket_with_options_async(
                batch_get_acp_connection_ticket_request,
                runtime
            )

            # 记录响应
            logger.debug(f"API response: {UtilClient.to_jsonstring(resp)}")

            # 检查响应状态
            if resp.status_code != 200:
                logger.error(f"API request failed with status code: {resp.status_code}")
                return None

            # 返回实例连接模型列表
            return resp.body.instance_connection_models

        except KeyError as e:
            logger.error(f"Missing environment variable: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get ACP connection ticket: {e}")
            return None

    async def describe_android_instances(self) -> Optional[List[DescribeAndroidInstancesResponseBodyInstanceModel]]:
        describe_android_instances_request = eds_aic_20230930_models.DescribeAndroidInstancesRequest(
            instance_group_id=os.environ['INSTANCE_GROUP_ID']
        )
        runtime = util_models.RuntimeOptions()
        try:
            # 复制代码运行请自行打印 API 的返回值
            resp = await self.client.describe_android_instances_with_options_async(describe_android_instances_request, runtime)
            # 记录响应
            logger.debug(f"API response: {UtilClient.to_jsonstring(resp)}")
            # 检查响应状态
            if resp.status_code != 200:
                logger.error(f"API request failed with status code: {resp.status_code}")
                return None

            # 返回实例连接模型列表
            return resp.body.instance_model
        except Exception as e:
            logger.error(f"Failed to describe Android instances: {e}")
            return None
