# 企业文档知识构建平台

当前代码覆盖两条连续主链：

```text
文件/压缩包/目录 → 预处理与区域识别 → DocumentIR
DocumentIR → 知识单元构建（语义树、知识单元、证据定位） → 混合检索 / 版本与增量管理 / 结构化压缩
```

项目只保留这一条生产路径，不包含旧 API、私有补丁或历史报告脚本。**知识单元构建**已真实实现，可在没有模型、GPU 和网络的环境中独立运行。后续功能统一使用“混合检索、质量评测、版本与增量管理、结构化压缩、检索问答服务”等功能名称，不再使用内部编号代替正式命名。

## 代码结构

```text
src/doc_parser/
  preparation.py          # 格式识别、原生抽取、Office/PDF/图片页包
  pipeline.py             # 区域对象、识别编排、DocumentIR 与 Markdown 导出
  routing.py              # native/light/VLM 分流与审计记录
  visual_gate.py          # 可解释的图像和版面风险特征
  model_backend.py        # 公开 PaddleOCR/PaddleX 模型接口与单页运行时
  runtime_pool.py         # 显存准入、页级并行、顺序回收
  orchestration.py        # 文件/文件夹入口、CPU 并行、失败隔离、知识单元主链接入
  document_ir.py          # 核心 IR 数据模型与校验
  knowledge_units.py      # 标题栈、题注合并、长度切分、位置表与硬校验
scripts/
  run_parser.py                # 解析生产 CLI，可直接生成知识单元
  build_knowledge_units.py     # 已有 DocumentIR 的独立/批量知识单元 CLI
  light_ocr_worker.py          # 常驻轻量 OCR 子进程
  benchmark_parser.py          # 预热、重复计时、源码快照与实验清单
config/
  parser.json             # 路由、识别和并行参数
  knowledge_units.json    # 知识单元切分与复核参数
tests/                        # 无模型单元与 CLI 端到端测试
```

## 安装与无模型预处理

Windows 可使用项目虚拟环境：

```bat
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\run_parser.py --input "待处理目录" --output "var\prepared" --prepare-only --cpu-workers 4
```

`--prepare-only` 不加载任何模型。它仍会生成页图、原生对象、页包和 DocumentIR。纯文本、JSON、DOCX、PPTX、XLSX 等可保留原生正文；只有视觉页骨架时还没有可构建的正文块。

## 完整识别并生成知识单元

先安装与本机 CUDA 匹配的 PaddlePaddle，再安装固定版本适配器依赖：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-paddleocr.txt
.venv\Scripts\python.exe scripts\run_parser.py --input "待处理目录" --output "var\parsed" --config config\parser.json --build-knowledge-units --knowledge-unit-config config\knowledge_units.json
```

默认模型目录：

```text
models/PP-DocLayoutV3_safetensors
models/PaddleOCR-VL-1.6
models/PP-OCRv6_small_det
models/PP-OCRv6_small_rec
```

设备与模型名通过 `.env.example` 中列出的环境变量配置。密钥和凭证不得放进配置或 Git。

## 从已有 DocumentIR 离线生成知识单元

单份 `document.json`、单文档 `results/` 目录均可：

```bat
.venv\Scripts\python.exe scripts\build_knowledge_units.py --input "var\parsed\results\document.json" --output "var\knowledge-units-one" --config config\knowledge_units.json
.venv\Scripts\python.exe scripts\build_knowledge_units.py --input "var\parsed\results" --output "var\knowledge-units-one"
```

递归处理解析批次，并使用纯 CPU 文档级并行：

```bat
.venv\Scripts\python.exe scripts\build_knowledge_units.py --input "var\parsed-batch" --output "var\knowledge-units-batch" --workers 4
```

批量模式只发现名为 `document.json` 的文件，保持源目录相对结构；坏 IR 会记录 `knowledge-units-error.json`，其他文档继续。输出目录必须为空，且不会把位于输入树内的输出再次当作输入。

## 知识单元规则与输出

- 标题栈形成章节容器；容器 `indexable=False`，正文单元才供检索索引。
- 表格、图片和公式保持原子单元，不因长度拆碎；相邻同页题注并入表格或图片。
- 普通段落和列表只在同章节、同页、同类型且长度合适时合并，不猜测跨页续段。
- 超长正文按标点或换行切分，无自然边界时才硬切，并记录 `length_fallback`。
- `unicode-word-v1` 是本地确定性长度单位，不冒充后续向量模型的 tokenizer。
- 同一源块可通过互不重叠的字符区间拆成多个单元；每个源块必须被位置表引用，或有明确忽略原因。

每份文档生成：

```text
knowledge-units/
  knowledge-units.jsonl      # 一行一个容器或正文单元
  element-locations.jsonl   # 单元到原始页、块、框和字符区间
  document-tree.json        # 语义父子树
  manifest.json             # 输入哈希、完整配置、覆盖统计和诊断
```

字段定义、ID 生成和一致性约束见 [知识单元构建与证据定位技术方案](docs/知识单元构建与证据定位技术方案.md)。

## 并行参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `cpu_workers` | 0 | 0 为自动，最多 4 个且不超过文件数；正整数为固定值 |
| `gpu_workers` | 0 | 0 为显存自动准入；正整数为固定运行时数 |
| `max_gpu_workers` | 4 | 自动模式的 GPU 运行时上限 |
| `gpu_free_margin_mb` | 4096 | 每次新增运行时必须额外保留的 MiB |
| `gpu_worker_floor_mb` | 2048 | 显存增量异常偏小时采用的最低估计 |

也可通过 CLI 覆盖：`--cpu-workers 4 --gpu-workers 0 --max-gpu-workers 3 --gpu-free-margin-mb 6144`。知识单元构建不复制模型，可随各文档的 CPU 任务并行。详细算法和限制见 [并行与实验说明](docs/并行与实验说明.md)。

## 完整输出

```text
输出目录/
  batch-manifest.json
  documents/<原相对路径及扩展名>/
    manifest.json
    native/
    pages/
    page_packages/
    results/
      document.json          # DocumentIR 主输入/输出
      document.md            # 阅读导出，不代替结构化 IR
      manifest.json          # 解析摘要及知识单元输出摘要
      routes.jsonl
      knowledge-units/       # 仅 --build-knowledge-units 时生成
        knowledge-units.jsonl
        element-locations.jsonl
        document-tree.json
        manifest.json
```

每个文件使用独立输出目录；不支持的格式记为 `unsupported`，坏文件记为 `failed`，其余文件继续处理。纯文本和纯原生文档不会触发模型加载。

## 基准实验

```bat
.venv\Scripts\python.exe scripts\benchmark_parser.py --input "待处理目录" --output "var\bench-001" --rounds 3 --warmups 1 --gpu-workers 0 --build-knowledge-units --knowledge-unit-config config\knowledge_units.json
```

实验目录包含逐轮记录、输入哈希、解析与知识单元完整配置、源码快照和 warm 轮中位数。指定 `--build-knowledge-units` 后，每轮计时也包含知识单元构建和四个文件写出。基准输出不能放在输入目录内，防止后续轮次把前轮产物当成输入。

## 验证状态与当前边界

- 35 项无模型测试覆盖知识单元核心规则、坏引用、字节级确定性、独立与批量 CLI、主链接入、核心管线、原生绕过、CPU 真并行、页级并发、显存准入和顺序回收。
- 已用真实 TXT 输入完成 `run_parser.py --build-knowledge-units` 端到端样例，并完成 1 轮预热、2 轮正式计时的基准冒烟验证。
- 当前环境没有模型权重，尚未执行真实 PP-DocLayoutV3/PaddleOCR-VL 推理；不能把单元测试当作精度、速度或显存验收。
- 表格若返回未转换的 OTSL，会原样保存在 IR，并在 Markdown 的 `<pre>` 中展示，不伪装成 HTML 表格。
- 知识单元首版不做跨页续段猜测、LLM 改写、向量化或数据库写入；切分默认值仍需通过质量评测中的检索对照实验调整。
- 当前交付是 CLI 与 Python 库入口，不包含线上 FastAPI 服务。

运行全部测试：

```bat
set PYTHONPATH=src
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
