# 航摄姿态插值服务（Attitude Interpolation Service）

把低频惯导（INS）姿态样本投影到相机曝光时刻。姿态按**单位四元数**解释，
分量顺序固定为 **[w, x, y, z]**；由于 `q` 与 `-q` 表示同一旋转，插值始终沿
**最短旋转弧**进行，返回序列按曝光顺序重新定号，保证相邻帧影像足迹连续。

## 快速开始

```bash
# 构建并启动服务（宿主机端口默认 8080，可用 HOST_PORT 覆盖）
docker compose up app
HOST_PORT=9000 docker compose up app

# 一次性验证：app 健康检查通过后自动运行 verify，结束后自行退出
docker compose up --exit-code-from verify verify
echo $?   # 0 = 全部通过
```

`verify` 是一次性服务：仅在 `app` 通过健康检查后启动，依次执行
**构建检查**（字节码编译 + 应用导入）、**代码测试**（pytest 套件）、
**API 冒烟**（插值精度/连续性与间隙拒绝等），随后以位掩码退出码汇总：

| 退出码位 | 含义 |
|---|---|
| 0 | 全部通过 |
| 1 | 构建/编译检查失败 |
| 2 | 单元/接口测试失败 |
| 4 | API 冒烟失败 |

本地开发（无需 Docker）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python verify.py
```

## API

### `POST /api/attitudes/interpolate`

请求体：

```json
{
  "samples": [
    {"t": 1700000000000000000, "q": [1.0, 0.0, 0.0, 0.0]},
    {"t": 1700000000000010000, "q": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]}
  ],
  "queries": [1700000000000002500, 1700000000000005000, 1700000000000007500],
  "max_gap_ns": 10000
}
```

| 字段 | 约束 |
|---|---|
| `samples` | 2–2000 个样本；`t` 为纳秒整数时间戳，**严格递增**；`q` 为 4 个有限数（w, x, y, z），整体非零（按单位四元数解释，自动归一化） |
| `queries` | 1–500 个纳秒整数时间戳，**严格递增**，落在 `[samples[0].t, samples[-1].t]` 闭区间内（可位于端点） |
| `max_gap_ns` | 非负整数；每个查询的**包围样本间隔**不得超过该上限 |

时间戳请使用 JSON 整数字面量（历元纳秒 ~1.7e18 超出 float64 精确整数
范围 2^53，浮点字面量会被拒绝以避免静默舍入）。

响应 `200`：

```json
{
  "attitudes": [
    {"t": 1700000000000002500, "q": [0.9807852804032305, 0.0, 0.0, 0.19509032201612828]}
  ]
}
```

- 按查询顺序返回单位四元数（模长误差 ≤ 1e-9，插值误差 ≤ 1e-9）。
- **符号约定**：首项的第一个非零分量为正；后续项选取使与前项内积
  非负的等价符号（`q` / `-q`），保证相邻曝光姿态连续、无正负翻转。

### 错误响应（4xx，不产生部分结果）

任何输入越界、零四元数、非递增时间、180° 歧义或超长间隙都会返回
`400`，且可定位到具体索引：

```json
{
  "detail": {
    "code": "SAMPLE_GAP_EXCEEDED",
    "message": "queries[0]=... is enclosed by samples[0] (t=...) and samples[1] (t=...) whose gap 10000000 ns exceeds max_gap_ns=1000",
    "index": 0,
    "path": "queries[0]",
    "sample_index": 0,
    "gap_ns": 10000000,
    "max_gap_ns": 1000
  }
}
```

主要错误码：`SAMPLE_COUNT_OUT_OF_RANGE`、`QUERY_COUNT_OUT_OF_RANGE`、
`MISSING_FIELD`、`INVALID_TYPE`、`NON_INTEGER_TIMESTAMP`、
`TIMESTAMP_PRECISION_LOSS`、`NON_FINITE_COMPONENT`、`ZERO_QUATERNION`、
`NON_INCREASING_SAMPLE_TIME`、`NON_INCREASING_QUERY_TIME`、
`AMBIGUOUS_180_DEGREE_ROTATION`（相邻旋转恰为 180°，最短弧不唯一）、
`QUERY_OUT_OF_RANGE`、`NEGATIVE_MAX_GAP`、`SAMPLE_GAP_EXCEEDED`、
`INVALID_JSON` / `INVALID_BODY`。

### 逐行曝光（rolling shutter）

部分航摄相机逐行曝光，同一帧各行的真实采集时刻不同。请求体可选携带
`rolling_shutter` 对象：

```json
{
  "samples": [ ... ],
  "queries": [
    {"frameT": 1700000000000000000, "row": 0},
    {"frameT": 1700000000000000000, "row": 1}
  ],
  "max_gap_ns": 10000,
  "rolling_shutter": {
    "rowCount": 2000,
    "linePeriodNs": 2500,
    "direction": "top_to_bottom"
  }
}
```

| 字段 | 约束 |
|---|---|
| `rolling_shutter.rowCount` | 整数，2–20000（含端点） |
| `rolling_shutter.linePeriodNs` | 正整数；相邻两行曝光的时间间隔（纳秒） |
| `rolling_shutter.direction` | `"top_to_bottom"` 或 `"bottom_to_top"` |

启用后，`queries` 每项为 `{"frameT": <ns>, "row": k}`，其中 `row` 为
**零基**行号。曝光序号按读出方向换算：

- `top_to_bottom`：序号 = `row`
- `bottom_to_top`：序号 = `rowCount - 1 - row`

派生采集时刻为整数纳秒 `t = frameT + 序号 × linePeriodNs`。各项的派生
`t` 必须**严格递增**（与原整数 queries 相同的定序规则），并沿用原采样
范围、`max_gap_ns` 最大包围间隔、最短弧与连续定号规则。

响应仍为 `{"attitudes": [...]}`，每项按请求顺序返回：

```json
{"frameT": 1700000000000000000, "row": 0, "t": 1700000000000000000,
 "q": [1.0, 0.0, 0.0, 0.0]}
```

即回显 `frameT`、`row`，并给出派生 `t` 与单位四元数。

逐行曝光模式的额外错误码（均为 `400`，定位到查询索引或配置字段，
不产生部分姿态）：`ROW_COUNT_OUT_OF_RANGE`（路径
`rolling_shutter.rowCount`）、`NON_POSITIVE_LINE_PERIOD`（路径
`rolling_shutter.linePeriodNs`）、`INVALID_DIRECTION`（路径
`rolling_shutter.direction`）、`ROW_OUT_OF_RANGE`（路径
`queries[i].row`）；派生时刻倒序、越出样本范围或落入超长间隙时分别
复用 `NON_INCREASING_QUERY_TIME`、`QUERY_OUT_OF_RANGE`、
`SAMPLE_GAP_EXCEEDED`，定位到 `queries[i]`，并在错误上下文中附带
`frameT`、`row`、`derived_t`。

**省略 `rolling_shutter`（或显式传 `null`）时**，`queries` 仍为整数
纳秒数组，响应项仍为 `{"t", "q"}`，请求、响应与错误语义与旧版完全
一致。

### `GET /health`

返回 `{"status": "ok"}`，供容器健康检查使用。

## 数值约定

- 输入四元数按 `math.hypot` 稳健归一化（极小/极大分量不溢出）。
- 相邻样本先做符号对齐（内积为负则取 `-q`），再按 SLERP 沿最短弧插值；
  相邻旋转 |dot| ≤ 1e-12 视为 180° 歧义并拒绝（该阈值远高于 float64
  归一化噪声 ~1e-16，远低于任何可用的小于 180° 间隔）。
- 小角度（dot > 1-1e-9）使用级数权重避免 `acos`/`sin` 相消，全角度
  范围内插值误差 ~1e-15，远优于 1e-9 的交付容差。
- 插值参数 `u = (t - t_i) / (t_{i+1} - t_i)` 由整数纳秒精确计算，
  历元级时间戳（~1.7e18 ns）不损失精度。

## 项目结构

```
app/
  attitude.py   # 核心：校验 + 最短弧 SLERP + 输出符号约定（纯标准库）
  main.py       # FastAPI HTTP 层
tests/
  test_attitude.py  # 核心数值与校验单测
  test_api.py       # API 接口测试
verify.py           # 一次性验证：构建 + 测试 + 冒烟，位掩码退出码
Dockerfile
docker-compose.yml  # app（健康检查）+ verify（service_healthy 后运行）
requirements.txt
```
