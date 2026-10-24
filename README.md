# LLM API Resilience Lab

一个只用标准库的极简实验：当 LLM API 超时、限流、返回 5xx、Key 失效或响应格式不对时，客户端应该怎么处理，才不会把整段堆栈直接甩给用户。

## 为什么做这个

它来自一次真实的测试复盘。

在另一个「企业知识库问答助手」项目里，我故意保留了一个脆弱点：调用 API 失败时，`urllib` 的异常没有被接住，用户会直接看到 `HTTPError` 或 `URLError` 堆栈。这个仓库把「错误处理」单独拆出来，做成一个可读、可测试的小客户端。

不是生产级 SDK，也不追求覆盖所有协议细节。它只回答四个问题：

- 哪些错误值得重试？
- 哪些错误重试也没用？
- 退避多长时间？
- 最终给用户看什么？

## 它做什么

- 只依赖 Python 标准库，不需要 API Key 就能跑测试。
- 对错误分类：配置错误、空输入、401/403、400、429、5xx、超时、网络错误、响应格式错误。
- 只对临时性错误重试：429、5xx、超时、网络错误、无法解析的响应。
- 不对认证失败、请求被拒、响应结构缺字段做无意义重试。
- 指数退避，`RetryPolicy` 可配置次数、初始延迟和上限。
- 提供 `user_message(error)`，把异常映射成一句人话，不泄漏内部堆栈。
- 用本地 mock server 覆盖失败路径，测试可重复、零网络依赖。

## 错误分类

| 情况 | 异常 | 是否重试 | 给用户的提示 |
| --- | --- | --- | --- |
| URL/Key/模型名为空 | `ConfigurationError` | 否 | 检查配置 |
| 问题为空 | `ValidationError` | 否 | 请输入非空问题 |
| 401 / 403 | `AuthenticationError` | 否 | 检查 API Key 或权限 |
| 400 | `BadRequestError` | 否 | 检查输入、模型名或参数 |
| 429 | `RateLimitError` | 是 | 稍后再试或降低并发 |
| 500 / 502 / 503 | `ServerError` | 是 | 稍后重试或查看服务状态 |
| 连接超时 | `LLMTimeoutError` | 是 | 调大 timeout 或稍后再试 |
| 连不上 / 连接中断 | `NetworkError` | 是 | 检查网络或 API 地址 |
| 返回的不是 JSON | `ResponseError` | 是 | 响应不可用，稍后再试 |
| JSON 缺少 `choices/message/content` | `ResponseError` | 否 | 响应格式不符合预期 |

## 快速开始

要求 Python 3.8+，不需要安装任何第三方包。

```bash
python -m unittest -v
```

当前结果：

```text
Ran 11 tests in 0.711s

OK
```

测试会自己启动一个临时 HTTP 服务，所有请求都发往 `127.0.0.1`，不会调用真实模型。

## 目录

```text
llm-api-resilience-lab/
├── resilient_client.py       # 客户端、异常类型、重试策略、用户提示
├── mock_server.py            # 本地 mock，按 URL 路径模拟不同失败
├── test_resilient_client.py  # 11 个 unittest 用例
├── .github/workflows/tests.yml
├── .gitignore
└── README.md
```

## 用法示例

```python
from resilient_client import (
    ResilientLLMClient,
    RetryPolicy,
    LLMError,
    user_message,
)

client = ResilientLLMClient(
    api_url="https://api.example.com/v1/chat/completions",
    api_key="your-key",
    model="your-model",
    timeout=15,
    policy=RetryPolicy(max_attempts=3, base_delay=0.5, max_delay=4.0),
)

try:
    answer = client.chat("报销需要哪些材料？")
    print(answer)
except LLMError as exc:
    # 日志里可以保留 exc.status / exc.attempts / exc.retryable；
    # 给用户只显示一句可读的提示。
    print(user_message(exc))
```

## 测试覆盖了什么

`mock_server.py` 用不同路径模拟：

- `/ok`：正常返回
- `/401`、`/400`：不可重试的客户端错误
- `/429`、`/500`：可重试的临时错误
- `/flaky`：第一次 500，第二次成功
- `/slow`：响应超过客户端 timeout
- `/bad-json`：返回非 JSON
- `/missing-choices`：JSON 结构缺字段

`test_resilient_client.py` 断言的不只是「有没有抛异常」，还包括：

- `attempts` 是否是 1 或 2；
- `retryable` 是否符合预期；
- 退避序列是否是 `[0.01]`；
- 失败两次后是否只 sleep 一次；
- 认证失败是否真的没有重试；
- 用户提示是否没有泄漏原始异常文本。

## 已知局限

- 没有读 `Retry-After`，429 的等待时间由本地策略决定。
- 没有 jitter，多个客户端同时重试时可能仍然一起打满。
- 没有熔断器和请求队列，适合演示，不适合高并发生产。
- mock server 只模拟 HTTP 层；真实供应商的响应格式可能略有差异。
- 只支持 chat-completions 风格的一个最小请求体。

## 和另一个项目的关系

另一个仓库 `enterprise-kb-qa-agent` 故意保留缺陷，用来练习测试和复盘；这个仓库则把其中一个缺陷（API 异常直接冒泡）拿出来，做成可复用的错误处理层。

一个负责「暴露问题」，一个负责「修一个问题」。两个都是真实写出来、跑过测试的代码，不是包装出来的项目名。
