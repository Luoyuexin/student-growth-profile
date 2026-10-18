# StuProfile · 大学生成长智能画像平台（优化版）

> 萌芽杯 B 类作品　|　成员：罗粤馨、张家鸣、丰书毅、刘锶玛、商妍茹　|　指导老师：李大字、马昕、孙鸿洋
>
> 本文件夹为对照《StuProfile项目作品书》优化后的版本，逐项改动与判断说明见 **[优化改动说明.md](优化改动说明.md)**；原始代码另有完整备份，未做任何改动。

一个**本地化部署**、可实时调用云端大模型 API 进行数据分析与 AI 解读，并呈现学生个人 / 班级成长画像报告的动态网页应用。
所有数据分析和可视化图表均由后端动态计算与生成，而非前端写死的静态数据。

## 技术栈

| 层次 | 技术 |
|------|------|
| 开发语言 | Python 3.14 |
| 后端 / 数据分析 | Python + Pandas + NumPy + Scikit-learn（K-means） |
| AI 分析（创新点） | 云端大语言模型（OpenAI 兼容接口：DeepSeek / 通义千问 / 智谱 GLM 等） |
| 前端 | Streamlit（双端：学生端 / 教师端） |
| 数据库 | SQLite（本地轻量存储） |
| 可视化 | Plotly（交互图表）+ ReportLab（PDF 报告） |
| 安全 | 字段脱敏 + 分级权限 + 本地文件隔离 + 云端脱敏网关 |

## 目录结构

```
.
├── web_main/              # Streamlit 双端网页交互（学生端 + 教师端）
│   ├── app.py             # 入口：登录路由、数据自检、分析流水线触发
│   ├── student_app.py     # 学生端页面（画像 / AI 诊断 / 对话助手）
│   └── teacher_app.py     # 教师端页面（画像 / 名单 / 预警 / 学院研判 / 对话助手 / 配置）
├── analysis_core/         # 数据清洗、六维评分、K-means 聚类、风险判定
│   ├── data_clean.py      # 缺失值与异常值清洗
│   ├── normalize.py       # 特征无量纲归一化（Min-Max，方向统一）
│   ├── scoring.py         # 六维评分模型（二级指标→维度→综合，三级加权）
│   ├── kmeans_model.py    # 两级聚类：多维分型（肘部0.4+轮廓0.6自动定K）+ 一维四级分级（等级/边界/差距/类型命名）
│   ├── risk_level.py      # 风险等级判定
│   └── pipeline.py        # 分析流水线编排 + 个性化建议
├── report_module/         # 自动生成成长报告与可视化图表
│   ├── charts.py          # 雷达图 / 词云 / 分析树(Treemap) / 子维度柱状图 / 分布图
│   └── report_gen.py      # 学生 / 教师 PDF 报告导出（支持嵌入 AI 诊断）
├── ai_insight/            # AI 大模型分析（项目创新点）
│   ├── llm_client.py      # 云端 OpenAI 兼容 HTTP 客户端 + 脱敏网关 + 优雅降级
│   └── services.py        # F1~F5 五大 AI 功能
├── config_setting/        # 预警阈值、聚类参数、权限、提示词、AI 配置
│   ├── __init__.py        # 集中可调参数（支持 JSON 覆盖下发）
│   ├── prompts.yaml       # AI 提示词模板（已入库）
│   ├── llm_config.example.yaml  # AI 配置样例（含字段说明）
│   └── llm_config.yaml    # 实际配置（仅需填 api_key；含密钥，已 gitignore）
├── data_source/           # 示例学情数据集（CSV）+ 模拟数据生成脚本
│   └── generate_mock_data.py
├── local_database/        # SQLite 初始化与 ORM 操作
│   ├── db_init.py         # 建表、默认账号、连接管理
│   └── orm_ops.py         # 认证、读写（权限 + 脱敏）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 建议使用国内镜像加速：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2. 生成模拟数据（首次）

```bash
# 在项目根目录执行，会自动建库并写入模拟学情数据（含心理健康维度）
python data_source/generate_mock_data.py
```

也可以不手动执行——首次运行 `streamlit run` 时系统会自动播种数据并跑分析流水线。

### 3. 配置 AI 大模型（可选，创新点功能需要）

`config_setting/llm_config.yaml` 已预置好，默认对接**火山方舟 · Doubao-Seed-2.0-mini**（`doubao-seed-2-0-mini-260428`），**只需填入 API Key**：

1. 到火山方舟控制台（console.volcengine.com/ark）创建 API Key，并**在模型广场开通 `doubao-seed-2-0-mini-260428`**。
2. 把 Key 粘贴进 `config_setting/llm_config.yaml` 的 `api_key` 一行（`enabled` 已为 `true`）。
3. 验证连通性（用项目自带的检测函数，会直接告诉你失败原因）：

```bash
python -c "from ai_insight import llm_client as c; print(c.test_connection())"
```

返回 `(True, '调用成功 ｜ 模型 ... ｜ 返回：可用')` 即表示链路打通；若返回 `False`，第二项会写明是鉴权失败、模型未开通、限流还是网络问题，照提示排查即可。

- 接口为 **OpenAI 兼容**格式，可对接火山方舟、DeepSeek、通义千问、智谱 GLM、OpenAI 等；切换服务商只需改 `base_url` / `model` / `api_key`。
- 采用 **HTTP 直连**（`requests`，缺失时回退标准库 `urllib`），不依赖 `openai` SDK，避免 SDK 与 `httpx` 的版本冲突。
- `thinking` 字段按模型自适应：**GLM-5.3 / 5.3-Flash 这类推理模型的深度思考不可关闭**，传 `disabled` 会被服务端拒绝，因此对这类模型自动传 `enabled`；豆包等普通模型则传 `disabled` 以降低延迟。若某模型不认识该字段（HTTP 400），会自动去掉 `thinking` / `reasoning_effort` 重试一次。
- 推理型模型思考过程按输出 token 计费，可用 `reasoning_effort: low`（GLM-5.x 支持 low / high / max）平衡质量、延迟与成本。
- **隐私保障**：系统内置脱敏网关，向云端发送前会剥离姓名 / 学号等 PII，仅传维度分值与班级 / 年级聚合统计；`llm_config.yaml` 含密钥，已被 `.gitignore` 忽略，不会入库。

### 4. 启动应用

```bash
streamlit run web_main/app.py
```

浏览器打开终端输出的本地地址（默认 http://localhost:8501）。

### 5. 演示账号

| 端 | 账号 | 密码 |
|----|------|------|
| 学生端 | 任意学号（如 `202310100001`） | `stu123456` |
| 教师端 | `T2023001` / `T2023002` | `teacher123` |

> 提示：首次登录后点击侧边栏或登录页的「重新执行分析流水线」可刷新分析结果；
> 修改数据后建议重跑以保证画像最新。

## 核心功能

### 学生端
- 学号密码登录
- **顶栏切换**：综合总览页 + 六个维度分页面（学业发展 / 身心健康 / 社会参与 / 生活适应 / 品德素养 / 非认知能力）
- 个人综合评分、**综合等级**、**成长类型**与风险等级卡片
- 「**距上一综合等级还差 X 分**」与六维逐项「距上一等级差 X 分」（等级边界由一维 K-means 相邻质心中点确定，作品书 3.4.2 / 表 3-1）
- **六维能力雷达图**（叠加班级 / 年级均值参考线）
- **学生画像词云图**（自动生成特征标签）
- **维度分析树**（每个维度下钻二级指标，Treemap 矩形树图）
- 成长趋势折线图
- 多源数据来源标注（教务 / 一卡通 / 第二课堂 / 心理 / 学工等）
- 基于六维评分的个性化提升建议
- 🤖 **AI 成长诊断报告**（F1，自然语言 + 一键 PDF，量化结论强制 `[Fxx]` 引用 + 报告末尾附证据索引，作品书 4.4.2 三道防线）
- 🤖 **图表 AI 解读**（F3，雷达图 / 趋势图旁文字解读，结论可溯源）
- 🤖 **AI 对话助手**（F5，基于脱敏事实卡片问答，无引用数字句输出前剔除）
- 个人画像报告 PDF 导出（含 AI 诊断，ReportLab 排版，已注册中文字体，无浏览器依赖）

### 教师 / 辅导员端
- 工号密码登录与角色校验
- 班级整体画像分布（饼图 / 柱状图）+ 六维均值偏弱模块饼图
- 学生列表：搜索、筛选、排序，敏感信息脱敏显示，含**综合等级 / 成长类型**列
- 个体学生画像穿透查看（综合等级、成长类型、六维等级、确定性种子趋势图）
- 🔴 **高危学生实时弹窗预警**（D6，登录即提示）
- 高危 / 关注学生实时预警列表，支持标记备注
- 🏫 **学院学风宏观研判**（D3，年级 + 专业两级聚合：维度均值对比、风险分布）
- 🌳 **年级 / 班级整体分析树**（六维 Treemap 矩形树图，可单维查看或六维同显）
- 🤖 **AI 班级研判摘要**（F2）+ 动态偏弱模块饼图
- 🤖 **AI 帮扶话术**（F4，高危学生干预建议）
- 🤖 **AI 对话助手**（F5，班级视角）
- 批量 PDF 报告导出（含综合等级 / 成长类型列，中文字体正常显示）
- 配置管理：风险阈值可调、**各年级实时聚类模型查看**（自动选定的 K、成长类型名、一维等级边界均来自 `cluster_meta` 表）

### 配置管理
- 风险预警阈值可调（成绩、考勤、消费异常等）
- 各年级聚类 K 值与特征权重可调（`config_setting/__init__.py`）
- 教师管辖班级范围配置
- AI 模型开关与提示词模板（`config_setting/prompts.yaml`）

## 数据维度（六维一体）

- **学业发展（25%）**：GPA、挂科数、出勤率、自习时长、作业及时率、学分进度、学术竞赛
- **身心健康（20%）**：心理测评、压力指数、体测成绩、睡眠规律、情绪状态
- **社会参与（12%）**：第二课堂、志愿服务、社团任职、人际交往、竞赛获奖
- **生活适应（10%）**：饮食规律、作息规律、消费健康、离校异常
- **品德素养（15%）**：思想政治、遵纪守法、诚信守信、集体意识
- **非认知能力（18%）**：恒毅力、自制力、投入度、后设认知、自我感知、学习动机
- **风险**：连续缺勤、夜间未归、消费骤减、成绩骤降、离校异常

> 评分口径：各二级指标归一化到 0~100 → 维度内加权得维度分 → 六维按权重加权得综合分。
> 指标、权重、阈值集中在 `config_setting/__init__.py` 的 `DIMENSION_WEIGHTS` / `DIMENSION_SUBSCORES` / `RiskThreshold`，便于调优。

## 隐私与安全

- 师生双端分别登录，分级数据查看权限。
- 敏感字段（姓名、学号）全程脱敏显示（`mask_name` / `mask_id`，两字姓名同样打码：`张三→张*`）。
- 学生只能查看本人数据，教师只能查看管辖班级数据。
- 所有数据本地存储（`local_database/student_growth.db`），不上传云端。
- **调用云端大模型时经脱敏网关**：白名单字段 + 原始明细大字段双重剥离，仅发送带编号的事实卡片（维度分值与班级/年级聚合统计）；回复中含数字但无 `[Fxx]` 引用的句子输出前剔除，量化结论可溯源；API Key 存本地 `llm_config.yaml`（已 gitignore），不进 git / 数据库。

## 扩展点（预留）

- `config_setting.load_config_from_json()`：预留教务系统 / 配置中心下发 JSON 配置。
- `data_source/` 与 `local_database/orm_ops.py`：可替换为对接教务系统、一卡通系统的真实数据接口。
- `analysis_core/pipeline.run_pipeline(records=...)`：支持直接传入外部记录进行实时分析。
- `ai_insight/llm_client.py`：抽象 LLM 接口，可扩展更多模型接入方式。

## 常见问题（FAQ）

**Q1：运行 `streamlit run` 后登录不进去？**
请确认已先执行 `python data_source/generate_mock_data.py` 生成数据；学号格式为纯数字（如 `202310100001`），密码 `stu123456`。

**Q2：AI 功能点不动 / 显示"未启用"？**
请检查 `config_setting/llm_config.yaml` 的 `api_key` 是否已填写（`enabled` 应为 `true`）。未配置或调用失败时系统会优雅降级（AI 诊断回退为规则式建议），且界面上会**直接显示失败原因**。

**Q3：调用云端模型时报错？**
界面提示与 `llm_client.get_last_error()` 会给出具体原因，常见几类：
- `HTTP 401 鉴权失败` —— API Key 无效或格式不对，重新复制；
- `HTTP 404 模型未开通` —— 方舟控制台里该模型尚未「开通」；
- `HTTP 429` —— 触发限流或免费额度用尽；
- `HTTP 400 请求参数被拒绝` —— 多为 `thinking` / `reasoning_effort` 字段不被该模型支持，客户端会自动去掉这两个字段重试一次。
另需确认 `base_url` / `model`（用模型广场里的 Model ID）填写正确，且网络能访问对应服务商。

**Q4：想重置所有数据？**
删除 `local_database/student_growth.db` 与 `data_source/raw_student_data.csv`，重新执行第 2 步即可。

**Q5：想调整指标体系 / 权重 / 阈值？**
所有维度、二级指标、权重与风险阈值集中在 `config_setting/__init__.py`：
- `DIMENSION_WEIGHTS`：六维综合权重
- `DIMENSION_SUBSCORES`：每个维度下二级指标的字段、归一化区间与维度内权重
- `RiskThreshold`：风险触发阈值
修改后重新执行分析流水线即可生效，无需改动其他模块。

## 说明

- 模拟数据中的「成长趋势」为基于当前综合评分回溯生成的演示走势（原始数据为截面快照）；随机种子由学号 md5 派生，同一学生多次打开曲线一致。
- 密码哈希为演示用轻量实现，生产环境请替换为 bcrypt / argon2 并强制改密。
- 本项目全流程自主开发，核心代码由团队独立完成；前期训练仅使用公开脱敏数据集，不调取校内真实隐私数据。
