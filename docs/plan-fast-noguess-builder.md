# Plan: 快速无猜棋盘构建方案

> 状态: 待落地 (由其他 agent 实施，Claude 负责后续 review 验收)
> 作者: 多模型讨论收敛 (Gemini / GPT / DeepSeek / Claude)
> 目标: 解决高密度 (10×10/40、16×16) 下 no-guess 棋盘生成过慢的问题

---

## 0. 背景与问题

当前无猜棋盘生成链路在 [no_guess.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/data/no_guess.py) 中：

```
generate_no_guess_board()
  → ms_toollib.laymine_solvable()   # SAT-based 生成
  → is_solver_no_guess()            # 用 ProbabilitySolver 精确重验一遍
```

实测在 10×10/40 (40% 密度) 时已经很慢。但**没人量过到底慢在哪**：
- 可能是 `ms_toollib.laymine_solvable` 生成阶段慢；
- 可能是 `is_solver_no_guess()` 里 `ProbabilitySolver` 精确枚举 (见 [probability_solver.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/game/probability_solver.py#L17-L18) 的 `MAX_ENUM_CELLS=20`) 在高密度 frontier 上指数爆炸。

**核心纪律: 先量再动手。不要在不知道病根的情况下开新构造器。**

---

## 阶段一: Profiling (必做，零风险，无论结论都不白做)

### 目标
拆解 10×10/40 下生成单个无猜棋盘的耗时构成，定位真正瓶颈。

### 落地任务
新建 `scripts/profile_noguess.py`，要求:

1. 对固定参数 (默认 `width=10, height=10, mines=40`，可 CLI 覆盖) 生成 N 个棋盘 (默认 N=50)。
2. 分别累计并打点两段耗时:
   - `t_laymine`: 调用 `ms_toollib.laymine_solvable` 的总耗时。
   - `t_verify`: 调用 `is_solver_no_guess()` 的总耗时。
3. 统计:
   - 每段平均/中位/p95 耗时；
   - `ms_toollib` 调用成功率 (success=True 比例)；
   - `is_solver_no_guess` 通过率 (即 ms-toollib 声称 solvable 但本仓库 solver 拒绝的比例)；
   - 单个可用棋盘的平均端到端耗时 (含失败重试)。
4. 额外对比基线: 同样打点 8×8/10、8×8/32、10×10/40 三档，输出一张对比表。

### 实现要点 / 约束
- **不要修改** `no_guess.py` 的现有逻辑；profiling 脚本独立复制调用路径并插入 `time.perf_counter()` 打点。
- 复用 `generate_no_guess_board` 内部的调用方式 (参数顺序 `mt.laymine_solvable(height, width, total_mines, r, c, max_times=10000)`，注意 row/col 顺序)。
- 用 `argparse`: `--width --height --mines --n_samples --seed`。
- 输出 markdown 表格到 stdout，方便直接贴回讨论。

### 验收标准 (Claude review 时检查)
- [ ] 能跑出 `t_laymine` vs `t_verify` 的明确比例。
- [ ] 三档密度对比表完整。
- [ ] 不污染现有生成逻辑。

### 决策分叉
- **若瓶颈在 `t_verify` (重验慢)** → 进入阶段二-A (廉价验证降级)，**很可能阶段三的新构造器根本不需要**。
- **若瓶颈在 `t_laymine` (生成慢)** → 跳过阶段二-A，进入阶段三 (前向增量构造器)。

---

## 阶段二-A: 验证降级 (仅当瓶颈是重验时执行)

### 思路
`is_solver_no_guess()` 当前每步都跑 `ProbabilitySolver.compute_probabilities()` (精确枚举)。但判定"是否无猜"其实不需要精确边际概率，只需要判断"每一步是否存在可证明安全格"。`ConstraintSolver.find_safe_and_mines()` (纯约束传播，见 [solver.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/game/solver.py#L71-L96)) 快得多。

### 落地任务
1. 新增一个快速预筛函数 `is_constraint_no_guess(game)`: 仅用 `ConstraintSolver` 反复传播揭开，判断能否纯规则通关。
2. 改造验证链路为两级:
   - 先跑 `is_constraint_no_guess` (快) 做预筛；
   - **仅当**纯约束传播卡住 (找不到 safe，但还有 covered) 时，才 fallback 到 `ProbabilitySolver` 精确判定该步是否真有 `P(mine)=0` 格。
3. **关键正确性约束**: 最终语义必须和现有 `is_solver_no_guess` **完全等价** — 即"每一步都存在 `ProbabilitySolver` 可证明的 `P(mine)=0` 格"。`ConstraintSolver` 只是快速路径，不能放松或收紧 no-guess 合同。

### 验收标准
- [ ] 在阶段一相同的样本集上，新旧验证函数对每个棋盘给出**完全一致**的 no-guess 判定 (写一个对拍测试，跑 ≥500 个棋盘 0 分歧)。
- [ ] 端到端生成提速有明确数字 (对比阶段一 baseline)。

---

## 阶段三: 前向增量构造器 (仅当瓶颈是生成时执行)

### 核心思想 (四模型收敛结论)
**不筛棋盘，而是让棋盘和它的无猜证明同时前向生长，永不回头重验。**

维持不变式: **当前已揭开状态下，`ConstraintSolver` 一定能找到至少一个可证明安全格。**

```
1. 从随机首点 + flood-fill 初始揭开区出发 (此时棋盘上还没有"确定的"雷布局)
2. 取一个 ConstraintSolver 可证明安全的 covered 格，揭开它
3. 为新暴露的数字约束，在邻域 covered 格中"惰性"分配雷:
   - 关键简化: 我们是构造者，雷位自己定，只需找到【任意一个】
     能让 frontier 继续推进的可行布雷，不需要枚举所有解。
   - 用一个轻量前向 CSP: 在 frontier covered cells 上找一个布雷赋值，
     满足所有已暴露数字约束 + 全局剩余雷预算，且揭开后仍存在可证明安全格。
4. 若某步找不到可行赋值 → 局部 backtrack 上一个布雷决策 (不是重开整盘)
5. 雷数达标 + 全部安全格揭开 → 用 ProbabilitySolver 做一次【最终严格认证】
```

### 模型的角色 (明确降级为"可选剪枝器")
- **第一版不要接模型。** 先做纯 `ConstraintSolver` + CSP 基线。
- 仅当 backtrack 成为瓶颈时，再用 V5 模型的 `P(mine)` 热力图当**分支启发**: 优先在模型认为高概率是雷的格子布雷 (减少 backtrack)。
- 模型永远只是启发式，**绝不参与 no-guess 判定**。

### 落地任务 (分两步，先易后难)

**Step 3.1 — 无模型基线 (先在低密度跑通)**
- 新建 `scripts/incremental_builder.py`。
- 实现前向增量构造，复用 [game.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/game/game.py) 的 `MinesweeperGame` 和 [solver.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/game/solver.py) 的 `ConstraintSolver`。
- **先只在 8×8/10 验证概念跑通** (低密度 CSP 约束松，backtrack 少)。
- 产出的棋盘必须能通过 `is_solver_no_guess()` 严格认证。

**Step 3.2 — 上难度**
- 跑通 8×8/10 后，再冲 10×10/40 / 16×16。
- 若 backtrack 深度爆炸，再引入 V5 模型热力图剪枝 (复用 [inference.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/training/inference.py) 推理入口加载 `checkpoints/v5_replay_S5_denoise_rank_ft2/best_model.pt`)。

### 前向 CSP 实现提示
- frontier covered cells 通常只有几十个，规模可控。
- 复用 `build_constraints` / `normalize_constraints` ([solver.py](file:///Users/dengqichen/Downloads/minesweeper-transformer/src/game/solver.py#L18-L53)) 做约束剪枝。
- "只要一个可行解"目标下，用 DFS + 约束传播剪枝即可，不要做全枚举。

### 验收标准
- [ ] Step 3.1: 8×8/10 上稳定产出，100% 通过 `is_solver_no_guess`。
- [ ] Step 3.2: 10×10/40 上生成速度比现有方法快 ≥1 个数量级 (对比阶段一 baseline)。
- [ ] 所有产出棋盘 100% 通过 `ProbabilitySolver` 最终认证。

---

## 阶段四: 分布一致性验收 (任何路线都必做)

### 风险
任何构造式生成器产出的棋盘分布都 ≠ 均匀随机无猜分布。项目的训练数据和 100% 胜率 eval benchmark 全建立在现有 no-guess 分布上。`is_solver_no_guess` 只保证**正确性**，不保证**分布一致性**。

### 落地任务
新建 `scripts/compare_board_distribution.py`，对比【新生成器】vs【现有 ms-toollib 生成器】各 1000 个棋盘的统计量:
- 雷簇分布 (连通雷块大小直方图)；
- frontier 复杂度 (平均约束数 / 最大连通分量大小)；
- 平均求解步数 (`ProbabilitySolver` 通关所需步数)；
- 每步 ambiguous cell 数量分布。

### 验收标准
- [ ] 两个分布的关键统计量无系统性偏移 (输出对比表 + 简要结论)。
- [ ] 若有显著偏移，明确标注并评估对 benchmark 的影响。

---

## 总执行顺序 (给落地 agent)

1. **先做阶段一 profiling** → 把对比表贴回来，等 Claude 判断分叉。
2. 根据分叉结论，**只做** 阶段二-A **或** 阶段三 其中一条，不要两条都做。
3. 无论走哪条，最后都要做阶段四分布验收。
4. 每个阶段完成后产出可贴回的数字结果，Claude 逐项 review 验收标准。

## 不要做的事 (反模式)
- ❌ 不要从随机全局棋盘做拒绝采样 (用户已明确否决，高密度下成功率指数级低)。
- ❌ 不要让模型参与 no-guess 判定 (它有 1-2% 误差)。
- ❌ 不要在没跑 profiling 前就直接写新构造器。
- ❌ 不要同时实施阶段二-A 和阶段三。
- ❌ 不要修改现有 `no_guess.py` / `generator.py` 的语义 (新增旁路，不改老路径)。
