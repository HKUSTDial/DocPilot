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
- PDF 解析 Ready 后自动显示可开关的 OCR 类型高亮；文档下方可折叠查看 Markdown 与 JSON，标题栏支持格式选择、搜索、下载和复制。

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

## Flight Lab：沉浸式证据飞行

新增独立入口，不替换原有聊天界面，也不写入它的配置或会话历史：

- 开发模式：`http://localhost:5173/flight.html`。
- DataMosaic 一体化部署：`/analyze/flight.html`，重新执行根目录 `./start.sh` 构建后可用。
- 离线演示：`flight.html?demo=1`，点击 **Launch flight**；也可在页面中点击 **Preview a sample flight**。演示文档、数字和答案均为明确标注的虚构内容，不请求模型或读取 Workspace。

不再使用左侧 Library 或平铺缩略图：每个 PDF 对应一本立体装订书，多个文档以独立合拢书本排列在中央阅读台。当前文档以双页展开，带书脊、纸张厚度、缎带书签和双面 3D 翻页。可手动翻页或输入物理页码跳转；鼠标位于书本区域时，按住 **Ctrl + 滚轮** 可在 50%–200% 之间缩放书页内容，书封、书脊和外框尺寸保持不变。放大后可在每页内部滚动；聚焦书页时也可用 **Ctrl + 加号／减号** 缩放、**Ctrl + 0** 恢复原始比例。PDF 按实际显示尺寸重新高清渲染，证据框与飞机定位同步更新；书本以外不拦截浏览器快捷键。

实时模式默认只选择一本 PDF；**点击书本封面即打开并将该 PDF 设为唯一 QA 来源**，无需再逐本取消勾选。需要跨文档 QA 时，勾选其他书本下方的复选框。**Ask about** 显示实际发送给模型的文档范围，Launch 不会清空当前预览或跳回第一本书。导入文件后仅选择本次导入的 PDF；示例演示始终使用其固定虚构文档，与实时 QA 的选择独立。

Navigate 返回目标页后自动切换到对应书本并翻到目标双页；等待翻页及真实 PDF 页面渲染完成，飞机才飞向该页。导航页面标为 inspected/candidate；只有 `state.evidence_store` 中的证据才会高亮并送入右侧，最后呈现后端返回的答案。没有 bbox 时仅标记整页，不生成假的精准区域；答案引用的证据与其余收集证据分别标注。奇数总页数的最后一页旁仅显示装订衬页，不伪造 PDF 页面。

同一文档内保留原有翻页效果；跨文档时，飞机先停住，旧书两侧轻柔收拢并淡出，目标封面高亮，再展开新书并直接定位到目标物理页。**02 / LOG** 显示 `report.pdf · p.3 → other.pdf · p.12` 形式的来源与目标；新页实际渲染完成后飞机才重新落位、开始扫描。点击证据卡或封面换书也复用这一过渡，不增加模型请求。换书支持暂停与播放速度；Calm motion 或系统 reduced-motion 跳过收拢动画，但仍等待真实页面就绪。停止或重播会取消尚未完成的换书。

任务完成后小飞机回到阅读台的起飞位置。右侧顶部为 **02 / LOG**，显示当前事件编号和状态；下方 **03 / DISCOVERIES** 的卡片与书页高亮统一标为 **E1、E2、…**。点击卡片、编号或来源页码，会切换到对应文档、翻至证据页并滚动到高亮区域。当前选中的卡片与区域会加深强调；无精确 bbox 的证据保留整页虚线边框，不假定具体位置。定位与缩放不会重新请求模型，也不会改变 QA 文档选择。

飞机为约 40px 的简洁折纸造型，采用暖白纸面、青蓝折面、细折线与柔和投影。到达目标页后停在固定页边位置，等待 extract／focus 返回；同页证据扫描期间不挪动、不往返起飞点或证据栏，也不循环晃动。只有切换目标页时才移动，实际移动结束后隐藏风线与航迹。检查真实证据时保留邻页弱化、锁定脉冲、扫描光束和扫描线，但不再推拉书本镜头；新证据直接进入右侧卡片，不由飞机逐条搬运。最终仅为实际 supporting evidence IDs 绘制汇入答案的连接线。扫描效果不会提前生成证据卡或虚构 bbox，暂停和 reduced-motion 同时约束这些视觉效果。

右上角 **Day / Night** 开关支持日间暖白阅读台和夜间深蓝舞台。首次跟随系统外观，之后只在独立的 `docpilot-flight-theme-v1` 键中记录主题，不修改 Classic Chat 设置。提问后标题与封面区自动收紧，突出书本、飞机和右侧证据。

实时模式复用现有 default Workspace、上传接口及 `/api/chat/stream`，强制使用 DocPilot agent；每次提问使用独立的 `flight-*` 会话，不接触 Classic Chat 的会话记录。导入操作会向共享 Workspace 添加文件，但不会删除或移动原有文件。需要等待所选 PDF 解析 ready 后再启动；其他格式仍在原聊天页使用。

在 **Model settings** 中填写 provider、API base URL、model 和可选 API key。初次只读取原聊天页的模型名/地址作为默认值，不读取或持久化 API key，不修改原聊天页设置。支持 1×/2×/4× 动画速度、暂停/继续视图、停止请求、已有轨迹重播、证据定位和 reduced-motion 模式。**Pause view** 仅暂停动画，后端仍运行；**Stop flight** 才中断请求。重播不重新请求模型。动画可能稍晚于实时事件到达，不代表额外的 agent 动作。

视觉参考 [web-crawlers](https://github.com/cyohei9907/web-crawlers) 的“角色在真实内容上移动并留下反馈”的思路；本实现以独立 SVG 飞机、页面定位和真实证据事件驱动，不引入其代码、资源或新依赖。

轻量逻辑回归（在 `demo/chat/frontend` 执行）：

```bash
node_modules/.bin/tsc --target ES2021 --module commonjs --moduleResolution node --skipLibCheck --outDir /tmp/docpilot-flight-tests tests/flight.test.ts
node /tmp/docpilot-flight-tests/tests/flight.test.js
npm run build
```

`flight.html` 与 `index.html` 通过 Vite 多入口构建，独立加载组件和样式；后端无需改动。

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
