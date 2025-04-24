# 批量获取实例连接凭证完整工程示例

该项目为BatchGetAcpConnectionTicket的完整工程示例。

该示例**无法在线调试**，如需调试可下载到本地后替换 [AK](https://usercenter.console.aliyun.com/#/manage/ak) 以及参数后进行调试。

## 运行条件

- 下载并解压需要语言的代码;


- 在阿里云帐户中获取您的 [凭证](https://usercenter.console.aliyun.com/#/manage/ak) 并通过它替换下载后代码中的 ACCESS_KEY_ID 以及 ACCESS_KEY_SECRET;

- 执行对应语言的构建及运行语句

## 执行步骤

下载的代码包，在根据自己需要更改代码中的参数和 AK 以后，可以在**解压代码所在目录下**按如下的步骤执行：

- *Python 版本要求 Python3*
```sh
python3 setup.py install && python ./alibabacloud_sample/client.py
```
## 使用的 API

-  BatchGetAcpConnectionTicket：批量获取连接凭证。 更多信息可参考：[文档](https://next.api.aliyun.com/document/eds-aic/2023-09-30/BatchGetAcpConnectionTicket)

## API 返回示例

*实际输出结构可能稍有不同，属于正常返回；下列输出值仅作为参考，以实际调用为准*


- JSON 格式 
```js
{
  "RequestId": "7B9EFA4F-4305-5968-BAEE-BD8B8DE5****",
  "InstanceConnectionModels": [
    {
      "AppInstanceGroupId": "aig-1uzb6heg797z3****",
      "InstanceId": "acp-ajxvwo1u0hqvd****",
      "TaskStatus": "FINISHED",
      "TaskId": "cn-hangzhou@c9f5c2e8-f5c4-4b01-8602-000cae94****",
      "Ticket": "piVE58_AdmVSVW7SEW3*AE5*p8mmO5gvItsNOmv4S_f_cNpoU_BOTwChTBoNM1ZJeedfK9zxYnbN5hossqIZCr6t7SGxRigm2Cb4fGaCdBZWIzmgdHq6sXXZQg4KFWufyvpeV*0*Cm58slMT1tJw3****"
    }
  ]
}
```

