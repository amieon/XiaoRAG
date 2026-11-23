# 企业文档知识构建平台

当前代码覆盖两条连续主链：

```text
文件/压缩包/目录 → 预处理与区域识别 → DocumentIR
DocumentIR → 知识单元构建（语义树、知识单元、证据定位） → 混合检索 / 版本与增量管理 / 结构化压缩
```

项目只保留这一条生产路径，不包含旧 API、私有补丁或历史报告脚本。**知识单元构建**已真实实现，可在没有模型、GPU 和网络的环境中独立运行。后续功能统一使用“混合检索、质量评测、版本与增量管理、结构化压缩、检索问答服务”等功能名称，不再使用内部编号代替正式命名。

## 代码结构

```text
src/chapter_tree/
  reading_order.py        # 保留上游区域顺序并生成审计元数据，不做坐标重排
  context.py              # 构造前后 crop 原文窗口
  heading_classifier.py   # heading/body 数据链、Transformer 微调与推理
  native_classifier.py    # 完全信任 DOCX 原生标题证据的无模型模式
src/doc_parser/
  preparation.py          # 格式识别、原生抽取、Office/PDF/图片页包
  archive_security.py     # 压缩包安全边界、惰性 BFS 展开与来源链
  pipeline.py             # 区域对象、识别编排、crop 上下文、DocumentIR 与 Markdown 导出
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
  train_heading_classifier.py  # 按文档划分数据并微调标题二分类器
  classify_headings.py         # 离线预测 heading/body，不改写原始 IR
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

先安装与本机 CUDA 匹配的 PaddlePaddle，再安装固定版本适配器依赖。Windows/CUDA 12.9 的已验证示例为：

```bat
.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-paddleocr.txt
.venv\Scripts\python.exe scripts\run_parser.py --input "待处理目录" --output "var\parsed" --config config\parser.json --build-knowledge-units --knowledge-unit-config config\knowledge_units.json
```

布局模型与 VLM 默认使用 `paddle_dynamic`，轻量 OCR 使用 `paddle_static`。Torch 仅供图像预处理，不要为了 Transformers 推理改装 CUDA 版 Torch；混合加载 Paddle 与 CUDA Torch 曾在 Windows 上引发 cuDNN DLL 冲突。原生 Paddle/SentencePiece 加载器对中文绝对路径存在兼容性问题，默认模型路径使用项目根目录下的 ASCII 相对路径，因此完整识别命令应从项目根目录执行。

默认模型目录：

```text
models/PP-DocLayoutV3_safetensors
models/PaddleOCR-VL-1.6
models/PP-OCRv6_small_det
models/PP-OCRv6_small_rec
```

设备与模型名通过 `.env.example` 中列出的环境变量配置。密钥和凭证不得放进配置或 Git。

## 上游阅读顺序与 crop 上下文

视觉页直接使用上游版面模型返回的区域顺序：`region.index` 同时成为 `Block.reading_order`，不再运行 XY-Cut，也不根据 bbox 二次排序。页级 `metadata.reading_order` 记录 `model_provided / layout_region_order` 和有序块 ID，便于审计。

全部 crop 识别完成后，每个 `routes.jsonl` 记录及对应 IR 块的 `metadata.context` 都会保存同页的 `previous`、`next` 上下文。每项包含邻块 `block_id`、区域序号、原始布局标签、识别原文、字符区间、完整长度和截断标志；页首的前文、页尾的后文为 `null`，不跨页拼接。当前块文本仍使用原有 `result/text`，不在上下文里重复保存。

默认每侧最多保留 300 个 Unicode 字符，前一块取末尾、后一块取开头，可通过 `config/parser.json` 的 `crop_context_chars` 调整。截断只影响内嵌上下文窗口，不修改 IR 正文；下游可凭 `block_id` 回查完整邻块。

## 标题／正文二分类

标题／正文判定支持两种显式模式，不直接恢复完整章节树：

- `native`（默认）：完全信任 DOCX 的显式 Heading／Title 样式和 OOXML 大纲级别，有原生标题证据则为 `heading`，否则为 `body`。不加载模型，`confidence/scores` 为 `null`，不伪造概率。
- `transformer`：加载微调后的预训练 Transformer。模型输入为“当前块＋前一块＋后一块”，当前块置首以避免被右侧截断；没有训练产物时会明确失败，不会用随机初始化分类头生成结果。

DOCX 预处理会把原生标题证据保存到 `native/objects.json` 的 `structure`，再与 LibreOffice 渲染后的原生文本块对齐。只有规范化全文相同且匹配唯一时才传递证据；重复标题文本会标记 `ambiguous_exact_text`，不会任选一个。该证据随后进入 crop 的 `metadata.native_structures`。因此 native 模式是“信任已明确对齐的原生结构”，不是根据短文本或字体猜标题；未对齐的块按正文处理。

人工标签独立保存为 JSONL，每行一个块。`label` 可直接写 `heading/body`，也兼容 `section_heading/document_title/paragraph/text` 等等价名称：

```jsonl
{"block_id":"doc_xxx_p0001_b0000","label":"heading"}
{"block_id":"doc_xxx_p0001_b0001","label":"body"}
```

从解析结果训练，输出必须是新目录：

```bat
.venv\Scripts\python.exe scripts\train_heading_classifier.py --input var\parsed --labels datasets\heading-labels.jsonl --output var\heading-model-exp01 --base-model hfl/chinese-roberta-wwm-ext --epochs 3 --batch-size 8
```

训练与验证按文档隔离；单文档数据会全部用于训练且没有验证集，不能据此报告泛化精度。训练环境需要 PyTorch、Transformers 和 safetensors，建议与 PaddleOCR 推理环境分开，按本机 CUDA 版本安装 PyTorch。`training_summary.json` 保存超参数、文档划分、类别权重及每轮标题类 F1／混淆矩阵。

离线预测生成独立 JSONL，不覆盖 IR。无需训练的原生模式：

```bat
.venv\Scripts\python.exe scripts\classify_headings.py --input var\parsed --mode native --output var\heading-native.jsonl
```

需要已微调模型的模式：

```bat
.venv\Scripts\python.exe scripts\classify_headings.py --input var\parsed --mode transformer --model var\heading-model-exp01 --output var\heading-predictions-exp01.jsonl
```

上游 `raw_label` 可以辅助预标注，但若直接把它当全部真值，模型最多学习复现上游判断；要纠正标题误判，仍需人工复核和独立测试文档。

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

### 压缩包展开与安全

ZIP/TAR 使用 `archive_security.py` 中的单一公开入口 `iter_archive_documents_safely()`。它按 BFS 惰性产出普通文件：发现文档后立即交给预处理，发现子压缩包则入队，不递归调用文档处理，也不等待整棵压缩树全部解压。DOCX/PPTX/XLSX 会先被识别为 Office 文档，不会误当普通 ZIP 展开。

压缩包根 `manifest.json` 使用扁平 `documents` 清单，不再嵌套 `children`。每个成员保留：

- `source_path`：从外层压缩包到当前文件的完整来源链；
- `archive_path`：包含当前文件的压缩包链；
- `relative_path`：当前文件在最内层压缩包中的相对路径；
- `path`：本成员的独立预处理产物目录。

安全边界拒绝绝对路径、`..` 路径穿越、软链接/硬链接、非普通文件、加密 ZIP 成员、重名成员和异常压缩比，并核对声明大小与实际写出字节。整次 BFS 共享文件数量、解压总字节和嵌套深度预算；默认上限分别为 10000 个成员、2 GiB、4 层。安全拒绝会使当前压缩包任务失败，不会把攻击样本静默记为普通不支持文档。加密 ZIP 明确报出 `Encrypted archive member is unsupported`，不提供密码遍历，也不在配置、日志或清单中保存密码。

## 基准实验

```bat
.venv\Scripts\python.exe scripts\benchmark_parser.py --input "待处理目录" --output "var\bench-001" --rounds 3 --warmups 1 --gpu-workers 0 --build-knowledge-units --knowledge-unit-config config\knowledge_units.json
```

实验目录包含逐轮记录、输入哈希、解析与知识单元完整配置、源码快照和 warm 轮中位数。指定 `--build-knowledge-units` 后，每轮计时也包含知识单元构建和四个文件写出。基准输出不能放在输入目录内，防止后续轮次把前轮产物当成输入。

## 验证状态与当前边界

- 53 项无模型测试覆盖知识单元核心规则、坏引用、字节级确定性、独立与批量 CLI、主链接入、压缩包 BFS 与攻击拦截、核心管线、单栏/多栏/跨栏/单图环绕阅读顺序、顺序异常回退、原生绕过、CPU 真并行、页级并发、显存准入和顺序回收。
- 已用真实 TXT 输入完成 `run_parser.py --build-knowledge-units` 端到端样例，并完成 1 轮预热、2 轮正式计时的基准冒烟验证。
- 已在 Windows、RTX 4070 Laptop 8GB 上完成单页真实模型冒烟：布局模型、VLM 和轻量 OCR 均成功加载，一页产生 8 条区域路由且全部由轻量 OCR 识别，阅读顺序状态为 `confirmed`。该次 VLM 只加载未参与识别；单页成功不等于多栏精度、整本速度或显存峰值已经验收。
- 表格若返回未转换的 OTSL，会原样保存在 IR，并在 Markdown 的 `<pre>` 中展示，不伪装成 HTML 表格。
- 知识单元首版不做跨页续段猜测、LLM 改写、向量化或数据库写入；切分默认值仍需通过质量评测中的检索对照实验调整。
- 当前交付是 CLI 与 Python 库入口，不包含线上 FastAPI 服务。

运行全部测试：

```bat
set PYTHONPATH=src
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
