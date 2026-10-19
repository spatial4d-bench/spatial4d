# Spatial4D Evaluation

Spatial4D 的多模态问答评测工具，支持视频/图像推理、18类任务评分及任务宏平均汇总。推理与评分可分别运行：已有模型回答时，无需再次加载媒体或调用被测模型。

- `run_inference.py`：读取QA、提前抽取视频帧，调用模型服务并保存回答。
- `score_results.py`：对已有回答进行规则评分，或使用可选的文本裁判辅助解析，输出各题型分数、任务宏平均及逐题诊断。

## 目录

- [安装](#安装)
- [数据准备](#数据准备)
- [运行评测](#运行评测)
- [评分指标](#评分指标)
- [结果与诊断](#结果与诊断)
- [代码结构](#代码结构)
- [测试](#测试)
- [参考](#参考)

## 安装

使用 Python 3.10 或更高版本，在仓库根目录执行命令。

仅对已有回答评分时，只需要Python标准库：

```bash
python score_results.py --help
```

需要运行视觉推理时，安装客户端依赖：

```bash
python -m pip install openai opencv-python numpy tqdm
```

模型需部署为可兼容当前请求参数的OpenAI风格Chat Completions服务。推理脚本不负责下载权重或启动服务；服务端需支持 `mm_processor_kwargs`、`top_k` 和 `chat_template_kwargs` 等参数。

视频帧以本地 `file://` URI发送，模型服务必须能读取抽帧缓存，且允许访问该路径。远端服务不能直接读取客户端本地文件；图片输入则使用base64 data URL。仅支持公网图片URL/base64的接口不能直接用于当前视频推理入口。

## 数据准备

本仓库不包含QA、媒体或模型权重，需要自行准备：

```text
data/
├── qa_file_shuffle/
│   ├── action_prediction.json
│   ├── object_counting.json
│   ├── route_plan.json
│   └── ...
└── media/
    ├── videos/
    └── images/
```

### QA格式

每个任务使用独立的 `.json` 文件，文件名为任务标识。支持对象数组或等长列式JSON；不直接读取JSONL或压缩包。对象数组示例：

```json
[
  {
    "id": "example-001",
    "question": "Which action will happen next?",
    "options": ["A. turn left", "B. turn right", "F. go forward"],
    "ground_truth": "F",
    "video_path": "videos/example.mp4",
    "image_path": null,
    "dataset": "example_dataset",
    "scene_name": "scene_001"
  }
]
```

| 字段 | 说明 |
|---|---|
| `id` | 必填；同一任务内唯一 |
| `question` | 必填；非空题干 |
| `ground_truth` | 必填；选择题字母/有序路径序列，或非负数值 |
| `options` | 选择题选项；支持字母到内容的字典、带标签的字符串列表或规范选项文本。缺省时可从题干的选项行提取；数值题应为空 |
| `video_path` | 视频路径；可为媒体根目录下的相对路径或绝对路径 |
| `image_path` | 单张图片路径或路径列表；未提供视频时使用图片 |
| `question_type` | 可选；如提供，须与文件名对应的任务一致 |
| `dataset` / `scene_name` | 可选；数据集与场景信息 |

列式JSON使用同名字段，每个字段为等长数组，例如 `{"id": ["example-001"], "question": ["..."], ...}`。每条QA至少需要可读取的视频或图片；同时提供时优先使用视频。

推理前会校验所有QA文件的结构、任务、GT和重复ID。媒体缺失或损坏时，该条记录为推理失败，不发送无视觉输入的请求。

## 运行评测

### 1. 生成模型回答

通过环境变量配置路径和模型服务：

```bash
export SPATIAL4D_QA_DIR=/path/to/qa_file_shuffle
export SPATIAL4D_MEDIA_DIR=/path/to/media
export SPATIAL4D_FRAME_CACHE_DIR=/path/to/frame_cache
export SPATIAL4D_RESULT_DIR=/path/to/results
export SPATIAL4D_API_BASE=http://127.0.0.1:8000/v1
export SPATIAL4D_MODEL=YOUR_MODEL_ID

python run_inference.py
```

| 环境变量 | 用途 |
|---|---|
| `SPATIAL4D_QA_DIR` | QA JSON目录 |
| `SPATIAL4D_MEDIA_DIR` | 视频/图片根目录 |
| `SPATIAL4D_FRAME_CACHE_DIR` | 抽帧缓存目录 |
| `SPATIAL4D_RESULT_DIR` | 模型回答输出目录 |
| `SPATIAL4D_API_BASE` | 模型服务API根地址，通常以 `/v1` 结尾 |
| `SPATIAL4D_MODEL` | 服务端注册的模型ID |
| `OPENAI_API_KEY` | 推理服务密钥；未设置时使用本地服务占位值 `EMPTY` |

PowerShell可使用 `$env:SPATIAL4D_QA_DIR='D:\data\qa_file_shuffle'` 的写法设置相应变量。

当前推理设置：视频均匀采样64帧，图像尺寸640×480，缓存为JPEG（质量70）；短视频重复采样真实帧。请求使用 `max_tokens=32768`、`temperature=0.7`、`top_p=0.8`、`top_k=20`、`presence_penalty=1.5`，并设置 `enable_thinking=False`。这些设置在推理脚本中配置，需确认服务端支持。

回答写入 `result_<任务>.json`。完整且与当前QA的ID、GT、条数一致的结果文件会跳过；不完整或不匹配的文件会报错，不会静默覆盖。结果按文件原子保存，不提供文件内逐条断点续跑。

### 2. 规则评分

```bash
python score_results.py \
  --result-dir /path/to/results \
  --qa-dir /path/to/qa_file_shuffle \
  --require-all-tasks \
  --rule-based \
  --output metrics_rules.json
```

规则模式不联网、不调用裁判。未提供专用裁判key时也默认使用规则评分；仅指定裁判模型名称不会启用付费调用。

### 3. 可选文本裁判

通过专用环境变量提供裁判key，避免将密钥写入命令历史或源码：

```bash
export SPATIAL4D_JUDGE_API_KEY=YOUR_JUDGE_API_KEY

python score_results.py \
  --result-dir /path/to/results \
  --qa-dir /path/to/qa_file_shuffle \
  --require-all-tasks \
  --judge-model gpt-4o \
  --judge-cache judge_cache.jsonl \
  --output metrics_judge.json
```

也支持 `--api-key` 或 `--judge-api-key`。显式参数优先于 `SPATIAL4D_JUDGE_API_KEY`；传空字符串可禁用环境中的key。评分器不会自动读取推理用的 `OPENAI_API_KEY`。`--rule-based` 可强制关闭裁判。

裁判只接收题干、选项和模型回答文本，不接收媒体或GT，也不负责重新解题：

- 选择题调用裁判匹配字母；路径题保留顺序与重复项。
- 数值题的完整有限数值字符串直接评分；其他格式由裁判抽取数值，并检查该数值能否在原回答中找到。
- 裁判返回 `Z` 视为未匹配并记0；API/抽取失败保留原因并记0，不回退为规则得分。
- 鉴权、模型或请求配置错误会中止评测；临时服务/网络错误按上限重试。调用可能产生费用。

启用裁判时，目录内所有输入及必需的题干/选项会在首个请求前完成预检。缓存可复用已完成的裁判响应；同一缓存文件不要被多个进程同时写入。

### 评分参数

| 参数 | 默认值 / 作用 |
|---|---|
| `--result-dir` | 结果目录；默认使用 `score_results.py` 中的 `RESULT_DIR` |
| `--qa-dir` | 可选QA目录，按任务+ID补齐题干和选项；不覆盖结果GT |
| `--output` | 可选JSON报告路径 |
| `--require-all-tasks` | 要求全部18类任务存在 |
| `--rule-based` | 强制离线规则评分 |
| `--api-key` / `--judge-api-key` | 显式指定裁判key |
| `--judge-model` | `SPATIAL4D_JUDGE_MODEL`，未设置时为 `gpt-4o` |
| `--judge-base-url` | `SPATIAL4D_JUDGE_BASE_URL`，未设置时为 `https://api.openai.com/v1` |
| `--judge-timeout` | 单次请求超时，默认60秒 |
| `--judge-max-attempts` | 每题最多尝试次数，包含首次请求，默认3次 |
| `--judge-qps` | 串行HTTP请求限速，默认每秒2次 |
| `--judge-max-tokens` | 裁判输出token上限，默认512 |
| `--judge-cache` | 可选JSONL响应缓存路径 |

远程裁判地址必须使用HTTPS；仅本地回环地址允许HTTP。报告和缓存不能覆盖输入结果/QA文件，二者也不能使用同一路径。

## 评分指标

### 任务与单题分数

| 题型 | 任务标识 | 指标 |
|---|---|---|
| 选择题 | `3d_grounding`, `action_prediction`, `action_recognition`, `affordance_det`, `app_order_reasoning`, `egocentric_reasoning`, `object_attr_est`, `phy_plau_reasoning`, `rel_dist_est`, `rel_ori_est`, `scene_classification`, `spatial_memory`, `state_change_det` | Accuracy |
| 有序路径题 | `route_plan` | 整个序列精确匹配；`AB`不等于`BA`，`BB`不等于`B` |
| 数值题 | `abs_dist_est`, `object_counting`, `object_size_est`, `room_size_estimation` | MRA |

选择题单题分数为0或1。数值题使用Decimal计算10个阈值下的Mean Relative Accuracy；对于GT `g > 0`：

```text
T = {0.50, 0.55, ..., 0.95}
MRA(p, g) = mean( |p - g| <= g * (1 - t) for t in T )
```

边界相等计为满足阈值；GT为0时仅预测0得1分。例如预测55、GT100，MRA为0.2。

规则解析支持明确答案范围、结构化答案、选项内容、部分Markdown/数词/单位等格式；不从冲突候选中按GT选答案，不取区间中点，不自动换算单位，不把一个动作复制成多个动作。路径题结合题干空位检查顺序及完整性。

### 任务宏平均

先合并同一任务在不同文件中的样本，求各任务的平均分，再对18个任务等权平均，最终以百分数输出。空答、推理失败、解析失败和裁判失败均保留在任务分母中，不按样本数做Micro加权。

不足18任务时，只输出实际任务的 `PARTIAL TASK MACRO`，并列出缺失任务；指定 `--require-all-tasks` 后会报错。空文件会提示并跳过；没有任何样本时不产生总分。

## 结果与诊断

评分输入为 `result_*.json` 对象数组，每条记录至少包含 `model_answer` 和 `ground_truth`。任务由 `question_type` 或规范结果文件名确定；`id` 用于去重和匹配QA上下文。推荐同时保存 `question`、`options`、`is_choice`、`inference_status` 和 `inference_error`。

终端打印各任务样本数、指标、分数及任务宏平均。`--output` 报告包含：

- `task_scores`：各任务样本数、百分制分数、失败与解析状态统计。
- `macro_task_score_percent`：任务宏平均；同时检查 `benchmark_complete` 和 `missing_tasks`。
- `cases`：逐题原始回答、GT、规范化预测、单题分数、`parse_status`、`parse_method`、`notes` 和 `failure_group`。
- `judge` / `new_model_calls`：本次运行的裁判请求、缓存命中、用量和配置；启用裁判时逐题保留提示词、响应及失败原因。
- `context_missing_cases` / `empty_files`：上下文缺失和空文件提示。

失败分类包括 `empty_answer`、`inference_failure`、`parsing_failure`、`judge_failure` 和 `answer_no_match`；`parsed` 仅表示解析成功，不代表答对。

## 代码结构

```text
spatial4d_eval/
├── run_inference.py       # QA与媒体推理入口
├── score_results.py       # 评分与终端汇总入口
├── spatial4d/
│   ├── __init__.py
│   ├── config.py          # 任务注册表、别名、版本及默认配置
│   ├── rules.py           # 答案解析、单题规则评分与MRA
│   ├── judge.py           # 裁判提示词、HTTP客户端及响应缓存
│   ├── data.py            # QA/结果加载、上下文及路径保护
│   ├── evaluation.py      # 单题准备、评分和任务聚合
│   └── cli.py             # 评分参数与报告展示
├── tests/                 # 构造数据和本地模拟接口测试
└── README.md
```

程序化调用可使用 `spatial4d.evaluation.score_record` 评分单题，使用 `aggregate_results` 聚合，或使用 `calculate_directory` 完成整个目录评测。需要自行控制单题生命周期时，使用 `prepare_record → process_results`。目录评测会统一预检后再调用裁判，单题接口只校验当前题目。

部署时请保留入口脚本与 `spatial4d/` 包的目录关系。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试仅需Python标准库，使用构造QA和本地回环HTTP服务，覆盖通用答案格式、路径题顺序/重复/完整性、数值与MRA、任务宏平均、裁判分流与失败、缓存、输入保护和评分CLI。无需GPU、真实模型或付费API key。

测试不代替实际模型服务的接口兼容性、媒体可见性、配额或输出质量验证。

## 参考

项目的单题处理/任务聚合组织方式及文档结构参考 [VSI-Bench / Thinking in Space](https://github.com/vision-x-nyu/thinking-in-space)。
