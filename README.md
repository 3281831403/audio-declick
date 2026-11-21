# audio-declick

## 题面

修复老录音的咔哒声后，接缝处又冒出一串新爆音：二阶差分方向写反，检测到的是平台而非突变；同簇不去重，一次咔哒被修十次；单点修复用前值填充，出现直流台阶；区间修复不保持端点连续，接缝新咔哒；修复段能量不约束，出现爆破音；`method` 未知静默当线性；越界索引不校验，崩溃；过密咔哒不报错，整体被削平；批量共用检测数组，轨道之间串；`threshold<=0` 不校验，全部被当咔哒。declick/repair.py 是核心，tests/test_repair.py 15个红。要修的是：二阶差分方向与同簇去重、单点两侧插值、区间端点连续与能量包络、`method` 校验、索引与密度校验、批量隔离、`threshold` 校验。

## 运行方式

在仓库根目录执行：

```powershell
cd D:\gsb-20261119\audio-declick
python -m pytest tests -q
```

初始状态必须恰好是：

```text
15 failed
```

且没有 collection error、import error 或 syntax error；15 个失败都应来自尚未实现的桩（`NotImplementedError`）。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只用 Python 标准库（`math`）与 pytest；离线运行，不使用随机数，不调用 `sleep`。
- 音频统一使用 `list[float]`，标称样本范围为 `-1.0..1.0`；数值断言容差为 `1e-6`。
- 8 个公开函数当前都是 `raise NotImplementedError` 的桩，契约写在各自的 docstring 中。
- `tests/test_repair.py` 是验收契约，**不可修改**；实现不得改变测试期望。
