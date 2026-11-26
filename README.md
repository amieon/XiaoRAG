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
  zero_shot_classifier.py # OpenAI 兼容零样本标题判别，原生标题优先且无需微调
  heading_hierarchy.py    # none／streaming／streaming-context／global 父子关系恢复
  tree.py                 # 原生／上游标题与显式编号构建确定性章节树
  groups.py               # 章节树转两层父子分块可消费的 flat groups
src/doc_chunking/
  cdc.py                  # 有界滚动哈希、内容定义候选点与自然边界吸附
  engine.py               # 章节树 groups → parent/child，CDC 映射原块及列表条款约束
  validation.py           # 父子关系、正文切片、来源与字符区间硬校验
src/hybrid_retrieval/
  adapter.py              # 混合检索输入：structural-chunk.v1 → 统一检索单元与 parent 库
  tokenizer.py            # 中文单字＋相邻二元词，英文／数字／版本号完整保留
  bm25.py                 # 可序列化的纯 Python Okapi BM25 索引与查询
  embedding.py            # sentence-transformers 编码与精确稠密 Top-K
  index.py                # 两路索引建库、清单校验与命中字段装配
  merge.py                # 多文档／双 producer 导出合并与全局父子关系校验
  pgvector_store.py       # 可选 pgvector dense 导入与精确余弦 Top-K
  milvus_store.py         # 可选 Milvus FLAT+COSINE collection 与 dense 查询
src/versioning/
  store.py                # 路径定文档、线性完整快照、内容寻址向量与增量复用
src/doc_parser/
  docx_native_ir.py         # DOCX 原生段落直读、块到段落下标映射与 Heading 样式回写
  preparation.py          # 格式识别、原生抽取、Office/PDF/图片页包
  docx_numbering.py       # OOXML 自动编号列表、层级与序号提取
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
  build_docx_heading_dataset.py # 无 OCR 的 DOCX 标题补充、章节树和新样式 DOCX 数据集
  build_chapter_tree.py        # DocumentIR 独立构建章节树和结构 groups
  build_chunks.py              # 生成 structural-chunk.v1 两层父子块并硬校验
  export_retrieval_units.py    # 统一双方切块输出，导出 M2 检索单元
  merge_retrieval_units.py     # 多文档 units/parents 合并为一份建库语料
  build_retrieval_index.py     # 同时构建稠密向量与 BM25 本地索引
  query_retrieval_index.py     # 独立执行 dense 或 BM25 Top-K 查询
  store_dense_pgvector.py      # 复用 embeddings.npy 导入 pgvector，不重新编码
  query_dense_pgvector.py      # 独立查询 pgvector dense Top-K
  store_dense_vector.py        # --backend pgvector|milvus 统一向量入库
  query_dense_vector.py        # --backend pgvector|milvus 统一 dense 查询
  publish_document_version.py  # 发布单文档线性版本并复用不变 child 的向量
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

TXT/CSV/Markdown 等纯文本按 BOM、UTF-8、GB18030 顺序严格解码，并在 `native/objects.json` 的 `encoding` 记录实际编码；未知编码仍明确失败，不使用替换字符制造乱码。旧 Office（`.doc/.ppt/.pps/.xls`）不是由 Python 库直接读取二进制正文，而是通过 LibreOffice 转 PDF，再抽取文字层并保留页图；`.pps` 是老 PowerPoint 放映格式。LibreOffice 控制台输出按原始字节捕获，再按本地编码、UTF-8、GB18030 解码，避免中文 Windows 的转换日志导致 `UnicodeDecodeError`。

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

标题／正文判定支持三种显式模式；分类器只输出判定，章节树由独立构建步骤恢复：

- `native`（默认）：完全信任 DOCX 的显式 Heading／Title 样式和 OOXML 大纲级别，有原生标题证据则为 `heading`，否则为 `body`。不加载模型，`confidence/scores` 为 `null`，不伪造概率。
- `zero-shot`：调用 OpenAI 兼容的通用指令模型，无需训练或微调；已有原生标题不发送给模型，模型只补充缺失证据。每次请求包含当前块、前后块和上游标签，模型只接收 `b0/b1/...` 临时代号，程序校验完整覆盖后映射回真实块 ID，避免抄写长哈希出错。缺块、重复、未知 ID 或非法层级均明确失败。
- `transformer`：加载微调后的预训练 Transformer。模型输入为“当前块＋前一块＋后一块”，当前块置首以避免被右侧截断；没有训练产物时会明确失败，不会用随机初始化分类头生成结果。

DOCX 预处理会把原生标题证据保存到 `native/objects.json` 的 `structure`，再与 LibreOffice 渲染后的原生文本块对齐。只有规范化全文相同且匹配唯一时才传递证据；重复标题文本会标记 `ambiguous_exact_text`，不会任选一个。该证据随后进入 crop 的 `metadata.native_structures`。因此 native 模式是“信任已明确对齐的原生结构”，不是根据短文本或字体猜标题；未对齐的块按正文处理。

DOCX 自动编号列表另从 OOXML `numPr/numbering.xml` 提取，保存到 `lists` 和 crop 的 `metadata.list_structures`，包括文档内 `num_id`、层级、序号、编号格式和渲染前缀。自动编号段落不会被编号标题规则提升为章节；`num_id` 只用于当前文档，不作为跨版本稳定身份。标题样式与列表编号同时存在时，标题角色优先。

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

无需微调的通用模型模式（PowerShell；地址按实际部署修改）：

```powershell
$env:HEADING_MODEL_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:HEADING_MODEL_NAME = '你的模型名'
$env:HEADING_MODEL_API_KEY = '你的密钥'
.venv\Scripts\python.exe scripts\classify_headings.py --input var\parsed --mode zero-shot --output var\heading-zero-shot-exp01.jsonl
```

API 密钥只能放当前终端或权限受限的本地 `.env`，不得写入命令行参数、脚本、清单或 Git。零样本结果中的 `confidence/scores` 仍为 `null`，不将模型自报数值当作校准概率。

默认每次请求设置 `max_tokens=4096`。思考模型若只返回 `reasoning_content` 而 `content` 为空，可提高 `--max-tokens 8192`，并在 Qwen 兼容服务上使用 `--disable-thinking`。对于忽略关闭思考开关的服务，程序会尝试读取 `reasoning_content`，但其中必须是合法预测 JSON，并继续通过块覆盖、短 ID 和层级校验；普通推理文字不会被接受。服务不支持 JSON 模式时使用 `--without-response-format`；若服务拒绝思考扩展字段，则去掉 `--disable-thinking`，不要混用失败实验的输出目录。

上游 `raw_label` 可以辅助预标注，但若直接把它当全部真值，模型最多学习复现上游判断；要纠正标题误判，仍需人工复核和独立测试文档。

## 无 OCR 的 DOCX 标题数据集生成

当原始 DOCX 没有 Heading 样式时，可以直接读取 OOXML 段落和表格，不运行 OCR／版面模型。通用模型只判断缺少原生证据的段落；已有 Heading 样式保持优先。标题识别与层级恢复是两个阶段：前者回答“是不是标题”，后者让模型选择 `parent_id`，程序统一计算 `heading_level=parent.level+1`，不再把局部批次猜测的绝对层级当成可靠全局结构。随后按原生段落下标回写 `Heading 1..6`，不会按重复文本猜段落，也不会覆盖原文件。

```powershell
$env:HEADING_MODEL_BASE_URL = 'http://127.0.0.1:8000/v1'
$env:HEADING_MODEL_NAME = '你的模型名'
$env:HEADING_MODEL_API_KEY = '你的密钥'
.venv\Scripts\python.exe scripts\build_docx_heading_dataset.py --input "datasets\raw-docx" --output "datasets\styled-exp01" --batch-size 8 --hierarchy-mode streaming
```

`--hierarchy-mode` 支持四种可对照实验的模式，模型名、服务地址和规模始终由参数或环境变量决定，代码不绑定 7B 或任何固定模型：

- `none`：不恢复父子关系，所有标题（包括原生 Heading 和明确编号标题）都平铺为一级，不调用层级模型，作为 baseline。
- `streaming`：逐个处理无锚点标题，只发送当前标题和“根到上一标题”的活动路径；模型只能选择 `root` 或路径内节点，天然拒绝未来引用和环。
- `streaming-context`：在活动路径之外增加当前标题前后文和最近最多三个已关闭标题，便于比较上下文是否值得额外输入。
- `global`：一次发送完整有序标题短清单，模型只为无锚点标题选择任意前序标题作为父节点；程序严格检查缺块、重复、未知 ID 和未来父节点。

原生 Heading／大纲级别与明确编号规则始终优先，不为这些标题调用第二次模型。每个标题的首个根节点由程序确定，不伪记为一次模型决策。各模式应分别使用新的输出目录，例如 `styled-none-exp01`、`styled-streaming-exp01`、`styled-context-exp01`、`styled-global-exp01`，不得混合两次实验产物。

这里的“原生结构”是 Heading／Title 样式及 OOXML `outlineLvl`，不是页面上可见的自动目录文本。可见目录可能过期或手工改错，而且重复进入正文会污染知识库，因此不作为最高优先级结构来源。

另提供完全无模型的纯原生 baseline：

```bat
.venv\Scripts\python.exe scripts\build_docx_heading_dataset.py --input "datasets\raw-docx" --output "datasets\native-only-exp01" --native-only
```

`--native-only` 不需要模型地址、模型名或密钥，只把显式原生标题纳入章节树并保留其原生层级；普通段落即使写成“第一章／1.1”也不会被提升为标题。它与 `none` 的区别是：`native-only` 检验“原生结构本身能恢复多少”，`none` 检验“所有标题完全平铺时会怎样”。纯原生结果在 manifest 中记录 `heading_mode=native`、`hierarchy_mode=native`、`model=null`。

每次实验使用新的空输出目录。每个文档生成：

```text
documents/<原文档名>/
  document.json              # 逐段／逐表 DocumentIR，保留 native_id 和段落下标
  heading-predictions.jsonl  # 标题判定、真实父块 ID、最终层级与各自来源
  chapter-tree.json          # 可直接查看的章节树、标题路径和判定来源
  chapter-groups.json        # 后续 parent/child 切块输入
  styled.docx                # 带 Heading 样式的新文件，可在 Word 导航窗格查看
```

`heading-predictions.jsonl` 中，`hierarchy_parent_id` 是真实父块 ID，`hierarchy_source` 区分 `native`、`explicit_numbering`、`document_root`、各模型模式和 `baseline_no_tree`。`dataset-manifest.json` 汇总 `hierarchy_mode`、实际层级模型决策数、树最大深度、章节数、原生／模型标题数、警告数和新 DOCX 路径。这里没有自动插入可见目录，避免目录文本与正文重复进入知识库；如果需要 Word 目录，可在 `styled.docx` 中通过“引用 → 目录”生成。真实模型质量仍需在独立人工复核集上评估，单元测试中的假响应只验证数据流、格式校验和回写正确性。

继续生成两层父子块时，必须把同一份预测显式传给切块器：

```bat
.venv\Scripts\python.exe scripts\build_chunks.py --input "datasets\styled-exp01\documents\demo\document.json" --predictions "datasets\styled-exp01\documents\demo\heading-predictions.jsonl" --mode hybrid --output "var\chunks-exp01"
```

重放 `none` 实验时，独立建树和切块都额外传入 `--flat-headings`，保证下游不会从原生样式重新恢复旧层级：

```bat
.venv\Scripts\python.exe scripts\build_chapter_tree.py --input "datasets\styled-none-exp01\documents\demo\document.json" --predictions "datasets\styled-none-exp01\documents\demo\heading-predictions.jsonl" --mode hybrid --flat-headings --output "var\tree-none-exp01"
.venv\Scripts\python.exe scripts\build_chunks.py --input "datasets\styled-none-exp01\documents\demo\document.json" --predictions "datasets\styled-none-exp01\documents\demo\heading-predictions.jsonl" --mode hybrid --flat-headings --output "var\chunks-none-exp01"
```

## 确定性章节树与结构 groups

章节树按 `page_no/reading_order` 的物理顺序构建，不用坐标二次排序，也不把树遍历顺序当成正文顺序。标题证据优先级为原生 DOCX 结构、可选 Transformer 预测、上游布局标题；普通段落中的显式“第一章／1.1”编号可保守补充层级。列表、表格、公式和带句末标点的编号正文不会被编号规则提升为标题。完全没有标题时保留根节点下平铺，层级跳跃只记录警告，不伪造缺失章节。

```bat
.venv\Scripts\python.exe scripts\build_chapter_tree.py --input "var\parsed\results\document.json" --output "var\chapter-tree-exp01" --mode native
.venv\Scripts\python.exe scripts\build_chapter_tree.py --input "var\parsed\results\document.json" --output "var\chapter-tree-exp02" --mode hybrid --predictions "var\heading-predictions-exp01.jsonl"
```

`--mode native` 只信任原生标题；`upstream` 使用上游标签和显式编号；`hybrid` 合并三类证据。输出目录必须为空，且不修改 `document.json`。每份输出包含：

- `chapter-tree.json`：`chapter-tree.v1`，保存任意深度节点、标题路径、块归属、判定来源、层级跳跃警告和覆盖自检。
- `chapter-groups.json`：`chapter-groups.v1`，保存 `strategy=chapter_tree` 的扁平结构 groups。空祖先标题并入首个有正文的后代，尾部纯标题保留为显式 heading-only group；每个输入块恰好覆盖一次。后续长度切分仍只生成 parent/child 两层，不增加孙块。

## 章节树生成两层父子块

普通文档不再对整篇执行 `clause/section/item/paragraph` 四选一：章节树决定 group 和完整 parent。连续普通正文不再按 800 字贪心错位，而是整体运行 `cdc-natural-text-v1`：32 个 Unicode 字符的有界滚动哈希产生内容定义候选点，再向前后 64 字符内的自然边界吸附。默认最小／目标／最大长度为 400／800／1600 Unicode 字符；候选点不保证恰好间隔 800 字，1600 是没有合适边界时的普通正文硬上限。

自然边界依次为空行、句末标点、单个换行、逗号／顿号／冒号；英文句点只有伴随空白或文本结尾时才作为句末，`v2.1`、`3.14` 不会被当成句号。达到最大长度仍无候选点时，从目标到上限之间选择最后一个最强自然边界，完全没有才硬切并记录 `cdc_hard_character_boundary`。CDC 每版独立重算，不读取旧切点，也不保存滚动哈希历史。

DOCX 自动列表项及显式续段、表格、公式和代码保持原子单元；明确的“第一条／第二条”仍是 child 的优先边界和 `clause_path`，不额外伪造章节或 parent。章节标题跟随其下第一个条款，不单独形成标题 child。CDC 区间会映射回一个或多个原 IR 块，保留 `logical_text_span` 和证据字符区间；标点和原始正文均不丢弃。

```bat
.venv\Scripts\python.exe scripts\build_chunks.py --input "var\parsed\results\document.json" --output "var\chunks-exp01" --mode native
```

输出 `chunks.jsonl`（全部 parent 在前、child 在后）、`chapter-tree.json`、`preprocessing.json`、`decision.json` 和 `self_check.json`。每个 group 必定有一个 parent 和至少一个 child；短组父子正文相同是正常行为。校验器会检查正文切片无缝无重叠、父子成员、来源定位、字符区间和 JSON Pointer。PPT／Excel 继续走显式格式通道，不套章节树。

## M2 统一知识单元导出

学长策略和章节树共用同一个适配器，不让向量化、版本管理或结构化压缩分别理解两套字段。适配器只消费单文档 `chunks.jsonl`，child 导出为 `retrieval-unit.v1`，parent 导出为独立的 `retrieval-parent.v1`；`embedding_text` 原等于 child 的 `text`，不去标题、不去重、不改写。`producer` 只作溯源，`content_hash` 供版本与增量管理使用，顺序 chunk ID 不作为跨版本稳定身份。

权限策略必须显式提供，例如 `security.json`：

```json
{
  "classification": "internal",
  "allowed_principals": ["group:engineering"]
}
```

执行导出：

```bat
.venv\Scripts\python.exe scripts\export_retrieval_units.py --input "var\chunks-exp01\chunks.jsonl" --security "security.json" --producer chapter_tree --output "var\retrieval-units-exp01"
```

学长输出使用同一命令，仅把 `--producer` 改为 `senior_strategy`。输出 `retrieval-units.jsonl`、`retrieval-parents.jsonl` 和 `manifest.json`；清单中的 `chunking` 保存完整切分算法、长度单位和参数，供版本快照审计。M2 只对前者的 `embedding_text` 向量化，命中后按命名空间内的 `parent_id` 取父块。适配器校验父子成员、重复 ID、空文本和来源定位；权限／密级缺失时失败关闭，不默认公开。

### 合并多文档建库语料

每份文档分别导出后，用合并器生成一份多文档语料，不需要手工拼接 JSONL：

```bat
.venv\Scripts\python.exe scripts\merge_retrieval_units.py --inputs "var\retrieval-units-docA" "var\retrieval-units-docB" "var\retrieval-units-docA-senior" --output "var\retrieval-merged-exp01"
```

输入顺序就是合并后的记录顺序，所有记录原样保留。同一文档可以同时包含 `chapter_tree` 和 `senior_strategy` 两套结果；同一 `producer+document_id` 重复输入会被拒绝，避免误把两个版本混入同一索引。合并器还检查输入清单与数量、记录 schema、document/producer 一致性、unit／parent 全局 ID 冲突、悬空 parent、跨文档错误父子引用和无 child 的 parent。全部校验通过后才开始写出，输出目录必须为空。

输出仍是 `retrieval-units.jsonl`、`retrieval-parents.jsonl` 和 `manifest.json`。清单记录每个输入的文档、producer、源 chunks 哈希和数量，以及总文档数、producer 分布和 `merge_complete=True`，可以直接作为下一步建库输入：

```bat
.venv\Scripts\python.exe scripts\build_retrieval_index.py --units "var\retrieval-merged-exp01\retrieval-units.jsonl" --parents "var\retrieval-merged-exp01\retrieval-parents.jsonl" --model "本地嵌入模型路径或模型ID" --device cpu --batch-size 16 --output "var\retrieval-index-exp01"
```

## 稠密向量与 BM25 建库查询

两路召回消费同一份 `retrieval-unit.v1`，但查询彼此独立，当前不做 RRF、权限过滤或 Agent 调用。先单独安装检索依赖；真实稠密检索需要 `sentence-transformers` 和可访问的嵌入模型，模型或依赖缺失会明确失败，不会用随机向量兜底：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-retrieval.txt
.venv\Scripts\python.exe scripts\build_retrieval_index.py --units "var\retrieval-units-exp01\retrieval-units.jsonl" --parents "var\retrieval-units-exp01\retrieval-parents.jsonl" --model "本地嵌入模型路径或模型ID" --device cpu --batch-size 16 --output "var\retrieval-index-exp01"
```

`--parents` 可省略，默认读取 units 同目录的 `retrieval-parents.jsonl`。输出目录必须为空。建库一次生成：

```text
embeddings.npy             # float32 向量矩阵，行序与 unit ID 文件一致
embedding-unit-ids.json   # 向量行到知识单元的显式映射
units.jsonl               # 命中后返回正文、来源与权限字段
parents.jsonl             # 独立 parent 库，不向量化
bm25-index.json           # 显式 postings、文档长度和 k1/b 参数
manifest.json             # 模型 ID、维度、分词器、数量和完成标志
```

生产编码使用模型归一化向量，点积等价于余弦相似度，再做精确 Top-K；当前不是近似向量数据库。BM25 使用 Okapi 公式，默认 `k1=1.5、b=0.75`，中文生成单字和相邻二元词，英文、数字、连接词与版本号保留完整 token。`v2.1` 这类版本号会作为一个 token，不拆成 `v` 和 `2.1`。

```bat
.venv\Scripts\python.exe scripts\query_retrieval_index.py --index "var\retrieval-index-exp01" --query "增量更新如何生效" --mode dense --top-k 5 --device cpu
.venv\Scripts\python.exe scripts\query_retrieval_index.py --index "var\retrieval-index-exp01" --query "增量更新如何生效" --mode bm25 --top-k 5
```

dense 查询从建库清单读取同一个嵌入模型；BM25 查询不加载嵌入模型。返回命中包含 `unit_id/score/rank`、child 文本、`parent_id`、标题路径、来源和权限字段，parent 正文不自动拼入结果。算法测试使用小型显式假向量验证矩阵、排序和数据流，不代表真实模型效果；真实模型的加载、精度和批量性能仍需使用实际模型与文档另行验证。

### 可选：只把 dense 存入 pgvector

方案中的 pgvector／Milvus 是向量层的备选实现，不要求把所有 JSON 搬进向量库。这里选择 pgvector：DocumentIR、chunks、parent、原始页图和 BM25 索引继续留在文件系统；PostgreSQL schema 只复制 dense 向量及命中返回所需的 ID、正文、来源和权限元数据。JSONL 仍是可重建的权威交换产物，数据库是派生索引。

先在 PostgreSQL 中安装并启用 pgvector：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

再安装 Python 驱动，在当前终端设置 DSN。不要把真实 DSN 写进脚本、README、清单或 Git；如果使用本地 `.env`，权限必须为仅当前用户可读写（Linux 为 600），且脚本不会自动加载该文件：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-pgvector.txt
set PGVECTOR_DSN=postgresql://用户名:密码@主机:5432/数据库名
.venv\Scripts\python.exe scripts\store_dense_pgvector.py --index "var\retrieval-index-exp01" --schema retrieval_exp01
```

导入直接读取已有 `embeddings.npy` 和 `units.jsonl`，不会重新运行嵌入模型。每个实验使用一个新的 schema，已存在则失败，不覆盖旧索引；建 schema、建表和批量写入在同一事务中完成。当前表保存 `unit_id/document_id/parent_id`、向量、child 正文、标题路径、来源、页码和权限字段，不保存 parent 全文或 BM25 postings。

```bat
.venv\Scripts\python.exe scripts\query_dense_pgvector.py --schema retrieval_exp01 --query "增量更新如何生效" --top-k 5 --device cpu
```

查询使用 pgvector 的 `<=>` 余弦距离排序，返回 `score=1-distance`，当前为精确扫描，没有创建 HNSW／IVFFlat 近似索引。数据库测试使用伪连接验证 SQL 数据流、字段恢复、模型一致性和 DSN 不泄漏；真实 PostgreSQL 服务、扩展安装和端到端性能尚需在本机联调。

### 可选：Milvus 后端与统一参数

pgvector 和 Milvus 可以共存，通过统一的 `--backend` 选择；两者都只是 dense 派生存储，不改变 JSONL、BM25 或 parent 库：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-milvus.txt
set MILVUS_URI=http://localhost:19530
set MILVUS_TOKEN=你的token
.venv\Scripts\python.exe scripts\store_dense_vector.py --index "var\retrieval-index-exp01" --backend milvus --name retrieval_exp01
.venv\Scripts\python.exe scripts\query_dense_vector.py --backend milvus --name retrieval_exp01 --query "增量更新如何生效" --top-k 5
```

无认证的本地开发服务可不设置 `MILVUS_TOKEN`。token 只从环境变量读取，URI、token 不写入返回清单。每个实验使用新 collection，已存在则失败；如果本次创建后写入失败，只清理本次新建的半成品 collection，不删除旧数据。

Milvus collection 使用显式 schema：`unit_id` 是主键，`embedding` 是固定维度向量，`embedding_model` 用于查询模型一致性检查，其余字段只包括 child 正文、parent ID、标题路径、来源、页码和权限元数据。向量索引为 `FLAT+COSINE`，因此当前是精确检索，不是 HNSW 近似召回；`score` 直接是 Milvus 返回的 COSINE 相似度，越大越相似。

Linux／macOS 也可安装本地文件库组件，Windows 原生不支持时应使用上面的服务端 URI：

```bat
.venv\Scripts\python.exe -m pip install -r requirements-milvus-lite.txt
set MILVUS_URI=var\milvus\retrieval.db
```

随后使用相同的 `store_dense_vector.py --backend milvus` 和 `query_dense_vector.py --backend milvus`。本地 `.db` 属于模型／实验产物，应放在 `var/` 等被 Git 忽略的目录。已用真实 Milvus Lite 文件库完成两条显式向量的建库、模型元数据读取和 Top-K 查询冒烟；这只验证数据库数据流，不代表真实嵌入模型效果，也不代表 Milvus 服务端部署与大规模性能已验收。

统一参数也适用于 pgvector：

```bat
.venv\Scripts\python.exe scripts\store_dense_vector.py --index "var\retrieval-index-exp01" --backend pgvector --name retrieval_exp01
.venv\Scripts\python.exe scripts\query_dense_vector.py --backend pgvector --name retrieval_exp01 --query "增量更新如何生效" --top-k 5
```

## 版本与增量管理：完整快照与向量复用

当前实现的是普通线性版本：同一规范化绝对路径对应同一个 `document_key`，源文件 SHA-256 变化时发布下一 `revision`，从 `r000000` 开始；路径和内容都未变则返回 `status=unchanged`，不创建版本，也不编码。移动或改名默认是新文档，后续若需保留身份应显式换绑，不能仅凭内容猜测。

每次发布消费一份单文档 `retrieval-preparation.v1` 导出，并保存完整源文件、全部 units／parents 和清单：

```bat
.venv\Scripts\python.exe scripts\publish_document_version.py --source "受管文档.docx" --input "var\retrieval-units-exp01" --store "var\knowledge-version-store" --model "本地嵌入模型路径或模型ID" --device cpu --batch-size 16
```

版本库结构如下：

```text
knowledge-version-store/
  vectors/
    vec_<模型与正文哈希>.npy     # 跨文档共享的不可变内容寻址向量
  documents/<document_key>/
    current.json                 # 原子替换的当前版本指针
    versions/r000000/
      source/original.docx
      retrieval-units.jsonl
      retrieval-parents.jsonl
      manifest.json
    versions/r000001/
      ...
```

parent 的 `parent_key` 来自 `producer + title_path + 同路径出现序号`，空标题路径记为 `<root>`；child 的 `unit_key` 只是 parent 内的顺序坐标。当前不把它当作可靠语义身份，也不做复杂单元匹配。向量复用只看 `content_hash + embedding_model`：正文相同且模型相同，即使 child 移动到其他 parent，也原样复用上一版的 `vector_ref`；正文变化、新增或换模型则生成新 `vector_id`。同版重复正文只编码一次，其他文档已写入全局向量池的相同正文也可复用。

`vector_ref` 显式记录后端、存储格式、模型、`vector_id` 和相对路径；版本只保存完整快照，不保存 diff、JSON Patch 或 `changes.json`，需要比较时现场计算。版本发布不会再次切块，而是消费 `build_chunks.py` 已经生成的 CDC children，并把导出清单中的 `chunking` 配置原样写入版本 manifest。因此局部文字修改后，远处内容相同的 child 可以继续按哈希复用向量。当前尚未实现分支 merge、Milvus／pgvector 的版本索引同步，也不把本地 NPY 向量池伪装成线上向量数据库。若编码或校验失败，不提交新版本；全局内容寻址向量可能留下未被版本引用的缓存，可按引用情况另行回收。

清单中的 `reused_from_previous_count` 和 `reused_from_global_count` 按 child 数统计，`newly_encoded_text_count` 按实际编码的去重正文数统计。源文件未变时直接沿用当前版本清单，因此即使命令指定另一个模型，也不会为相同源文件制造新版本。

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

- 当前全量回归为 148 项测试及 16 个子测试通过，覆盖四种层级模式、无模型纯原生 baseline、活动路径与上下文开关、原生／编号锚点优先、非法父节点拒绝、`none` 下游一致平铺、零样本标题输入输出校验、DOCX 原生段落映射与样式回写、知识单元、章节树、CDC 与自然边界、两层父子块、双生产者统一适配器、压缩包安全、核心管线与并行边界，以及 BM25 分词／评分／序列化、稠密 Top-K 排序、两路索引查询和版本发布／向量复用。
- CDC 冒烟样例将 80 个短句切成 14 个 child；只在开头编号增加一个字符后，第一个 child 改变，后续 13 个 child 正文完全不变。对应版本发布只编码 1 个新 child，其余 13 个 `vector_ref` 完全复用。该样例验证局部稳定性与数据流，不代表所有文档都只会改变一个块。
- 已用显式小型向量完成索引构建、BM25 独立查询及版本发布 CLI 冒烟，并用真实 Milvus Lite 文件库完成 FLAT+COSINE 建库、加载和 Top-K 查询；这些结果只验证算法和存储数据流。真实 `sentence-transformers` 模型尚未在本次环境中完成加载、检索精度或批量性能验证。
- 已用真实 TXT 输入完成 `run_parser.py --build-knowledge-units` 端到端样例，并完成 1 轮预热、2 轮正式计时的基准冒烟验证。
- 已在 Windows、RTX 4070 Laptop 8GB 上完成单页真实模型冒烟：布局模型、VLM 和轻量 OCR 均成功加载，一页产生 8 条区域路由且全部由轻量 OCR 识别，阅读顺序状态为 `confirmed`。该次 VLM 只加载未参与识别；单页成功不等于多栏精度、整本速度或显存峰值已经验收。
- 表格若返回未转换的 OTSL，会原样保存在 IR，并在 Markdown 的 `<pre>` 中展示，不伪装成 HTML 表格。
- 知识单元构建本身不做跨页续段猜测、LLM 改写或数据库写入；向量化与 pgvector 入库是后续独立步骤。切分默认值仍需通过质量评测中的检索对照实验调整。
- 当前交付是 CLI 与 Python 库入口，不包含线上 FastAPI 服务。

运行全部测试：

```bat
set PYTHONPATH=src
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
