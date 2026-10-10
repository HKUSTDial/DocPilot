# DocPilot Chat Demo

`demo/chat` 是 DocPilot 的交互式文档分析工作台，基于 React + FastAPI 构建，支持两类模型接入：

- `vLLM` 本地部署，默认按 `http://127.0.0.1:8000/v1` 访问。
- 自定义 OpenAI-compatible 大模型服务，只要提供 `base_url + api_key + model` 即可。

## 目录结构

```text
demo/chat
├── backend
│   ├── main.py
│   └── requirements.txt
├── frontend
│   ├── package.json
│   ├── src
│   └── vite.config.ts
├── workspace             # 运行时自动创建
├── README.md
└── start.sh
```

## 功能

- 三栏工作台前端
- 模型配置面板
- `vLLM / custom` 预设切换
- 远端模型列表发现
- 流式聊天输出
- 会话级文件上传、预览、删除、选中文档上下文

## 后端启动

```bash
python -m pip install -e .
cd demo/chat/backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8008
```

## 前端启动

```bash
cd demo/chat/frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5173
```

完成上述依赖安装后，也可以从项目根目录一键启动：

```bash
./demo/chat/start.sh
```

如果后端地址不是 `http://127.0.0.1:8008`，可以在前端启动前设置：

```bash
export VITE_API_BASE_URL=http://127.0.0.1:8008
```

## vLLM 示例

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

然后前端里填：

- `Provider`: `vLLM`
- `Base URL`: `http://127.0.0.1:8000/v1`
- `Model`: `Qwen/Qwen2.5-7B-Instruct`

## 说明

- Chat agent 会把选中的纯文本内容和 PDF 页面图像直接注入多模态模型请求。
- PDF 默认以 144 DPI 注入前 120 页；Default agent 仍使用 DocPilot 文档流水线。
- Default agent 直接复用项目根目录中的 `app/` 与 `pipelines/`；上传 PDF 后会调用 MinerU，并把运行产物写入 `demo/chat/backend/runs/`。
- 对话、上传文件、MinerU 结果和 workspace 状态均为本地运行产物，不包含在代码仓库中。
