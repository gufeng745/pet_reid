# 知识库问答 Agent (Qwen 版本)

基于阿里云通义千问 (Qwen) 的文档知识库问答系统。

## 目录结构

```
kb_agent/
├── agent.py              # 主程序
├── requirements.txt      # 依赖
├── .env                  # 配置文件
└── knowledge_base/       # 放置你的文档文件
    ├── doc1.txt
    ├── doc2.md
    ├── law.doc           # Word 文档 (.doc/.docx)
    └── ...
```

## 快速开始

### 1. 获取 API Key

访问阿里云 DashScope 控制台：https://dashscope.console.aliyun.com/apiKey

1. 登录/注册阿里云账号
2. 进入 API Key 管理页面
3. 创建新的 API Key

### 2. 配置 API Key

编辑 `.env` 文件：

```bash
OPENAI_API_KEY=sk-your-qwen-api-key-here
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL=qwen3.5-plus
```

### 3. 安装依赖

```bash
cd D:/claude_workspace/kb_agent
pip install -r requirements.txt
```

**注意**：Windows 系统会自动安装 `pywin32` 以支持 `.doc` 文件处理。

### 4. 准备知识库

将你的文档放入 `knowledge_base/` 目录，支持以下格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 纯文本 | .txt, .md | UTF-8 编码 |
| Word 文档 | .doc, .docx | Windows 需 pywin32，跨平台需 python-docx |
| PDF | .pdf | 需安装 pypdf |

已有示例文档：
- `employee_handbook.txt` - 员工手册
- `product_docs.txt` - 产品技术文档
- `law.doc` - 法律案件文档（中文）

### 5. 运行

```bash
python agent.py
```

## 使用示例

```
==================================================
知识库问答系统 - 按 Ctrl+C 或输入 'quit' 退出
==================================================

你：案件的原告是谁？

思考中...

助手：根据法院判决书，本案原告为：
- 原告 1：张某（案号 25127 号）
- 原告 2：李某（案号 25141 号）
两人共同起诉广发银行股份有限公司...

你：公司的年假政策是什么？

思考中...

助手：根据员工手册，公司年假政策如下：
- 入职满 1 年：5 天年假
- 入职满 3 年：10 天年假
- 入职满 5 年：15 天年假
...
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| OPENAI_API_KEY | API Key | 必填 |
| OPENAI_API_BASE | API 端点 | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| OPENAI_MODEL | 生成模型 | qwen3.5-plus |
| CHUNK_SIZE | 文本分块大小 | 500 |
| CHUNK_OVERLAP | 分块重叠 | 50 |

## 支持的模型

### 生成模型
- qwen-turbo
- qwen-plus
- qwen3.5-plus（推荐）
- qwen-max

## 费用参考

- Qwen 生成：约 0.008 元/千 tokens

详细价格请查看阿里云官网。

## 常见问题

**Q: API Key 无效？**
A: 确保在阿里云控制台正确创建并复制了 API Key

**Q: .doc 文件加载失败？**
A: Windows 系统需安装 `pip install pywin32`；跨平台建议转换为 .docx 格式

**Q: 中文乱码？**
A: 确保文本文件使用 UTF-8 编码保存；Word 文档由 Word 应用直接解析

**Q: 检索效果不好？**
A: 调整 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 参数，或确保文档内容清晰结构化
