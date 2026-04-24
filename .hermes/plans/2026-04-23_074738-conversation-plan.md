# keyhunt-bsgsBase58 迭代计划

## Goal

完成 TODO.md 中全部四项待办，将 keyhunt 推进到可发布状态：
1. BSGS 模式新 key 生成方式（目标 10x 提速）
2. 真正的 GPU/CUDA 支持（替换当前 placeholder）
3. 全模式回归测试套件（固定范围、可验证输出）
4. 补齐地址格式：BTC legacy P2PKH（已部分支持）、BTC bech32 P2WPKH、ETH（已部分支持）

---

## Current Context

- 主版本基于 `secp256k1/` 自定义大数库（已弃用 legacy 的 libgmp）
- 现有模式 8 个：`xpoint`, `address`, `bsgs`, `rmd160`, `pub2rmd`, `minikeys`, `vanity`, `wif-recovery`
- 当前 BTC 地址输出仅做 `pubkey -> rmd160 -> base58(P2PKH)`，无 bech32 路径
- CUDA 文件（`bsgs_cuda.cu`, `wif_recovery_cuda.cu`） kernel 为 stub，没有真正的 secp256k1 点乘逻辑
- 测试目录 `tests/` 只有手工输入文件，无自动化验证脚本
- Makefile 支持 `default` / `gpu` / `bsgsd` / `legacy` / `clean` 五个 target

---

## Proposed Approach

分四路并行推进，每路独立可验证，最后再整合：
- **Track A**：BSGS 数学层优化（纯 CPU，先不碰 CUDA）
- **Track B**：CUDA 内核真正落地（依赖 Track A 的数学实现可移植后接 GPU）
- **Track C**：地址格式补全（bech32 + ETH 完整链路 + 输入解析兼容）
- **Track D**：测试基础设施（pytest/shell 驱动，固定已知私钥范围，断言输出）

---

## Step-by-Step Plan

### Track A — BSGS 新 Key 生成（10x 提速）

| Step | Task | Files likely to change |
|------|------|------------------------|
| A1 | 阅读当前 `bsgsd.cpp` 的 `generatekey` / `bP` 表生成逻辑，确认瓶颈在随机 key 生成还是 baby-step 预计算 | `bsgsd.cpp`, `secp256k1/Random.cpp` |
| A2 | 引入 **deterministic stride + batch key derivation**：用 IntGroup 一次性预计算 1024/4096 个增量 key，减少重复点乘 | `secp256k1/IntGroup.cpp`, `bsgsd.cpp` |
| A3 | 实现 **endomorphism-aware batching**：在 BSGS 的 giant-step 循环里批量应用 β/λ 变换，复用中间结果 | `bsgsd.cpp`, `secp256k1/SECP256K1.cpp` |
| A4 | 添加 `-K` 参数让用户控制 batch size（默认 1024，可调） | `keyhunt.cpp`, `bsgsd.cpp` |
| A5 | 在 Puzzle 125/130 上测速，记录 `keys/s` 与内存占用基线，验证 ≥10x | `tests/125.txt`, `tests/130.txt` |

**验证**：`./keyhunt -m bsgs -f tests/125.txt -b 125 -q -s 10 -R -t 8` 提速前后对比。

### Track B — CUDA GPU 支持（替换 placeholder）

| Step | Task | Files likely to change |
|------|------|------------------------|
| B1 | 调研可用的 secp256k1 CUDA 方案：选项①移植 `secp256k1/Int.cpp` 到 device 代码；选项②引入 `secp256k1-cuda` 子模块；建议选①以保持代码统一 | `bsgs_cuda.cu`, `wif_recovery_cuda.cu` |
| B2 | 在 CUDA 侧实现 `uint256` 加减乘模逆（基于现有 `Int` 逻辑，去掉动态分配） | `bsgs_cuda.cu` |
| B3 | 实现 **fixed-window point multiplication** `k * G` kernel，每个线程处理一个 k 的搜索 | `bsgs_cuda.cu` |
| B4 | 实现 bloom filter / hash160 比对在 GPU shared memory 中的快速路径 | `bsgs_cuda.cu`, `bloom/bloom.cpp`（host 侧接口） |
| B5 | 在 `keyhunt.cpp` 中接通 `-g` / `--gpu` 开关，让 BSGS 和 address 模式都能调度 CUDA | `keyhunt.cpp`, `Makefile` |
| B6 | 添加 `make gpu` 的 CI 级编译检查（无 GPU 环境也能编译通过） | `Makefile` |

**验证**：有 NVIDIA 环境时运行 `./keyhunt -m bsgs -f tests/125.txt -b 125 --gpu -q` 应正确识别 CUDA 设备并输出搜索进度；无设备时优雅回退 CPU。

### Track C — 地址格式补全

| Step | Task | Files likely to change |
|------|------|------------------------|
| C1 | 梳理现有地址生成链路：`SECP256K1::GetHash160(P2PKH, ...)` 只出 rmd160，再由 `pubkeytopubaddress` base58 编码 | `keyhunt.cpp`, `secp256k1/SECP256K1.cpp` |
| C2 | 在 `SECP256K1.cpp` 新增 `GetHash160(P2WPKH, ...)`：先 SHA256 公钥，再取 20 字节作为 witness program | `secp256k1/SECP256K1.cpp`, `secp256k1/SECP256K1.h` |
| C3 | 新增 `bech32_encode()` 与 `bech32_decode()`（BCH 纠错码，hrp = "bc"），放在 `base58/` 或新建 `bech32/bech32.c` | 新建 `bech32/bech32.c`, `bech32/bech32.h` |
| C4 | 在 address / rmd160 / vanity 模式下识别目标地址前缀：`1` -> P2PKH, `3` -> P2SH（可略）, `bc1q` -> P2WPKH, `0x` -> ETH | `keyhunt.cpp` |
| C5 | ETH 链路确认：`generate_binaddress_eth` 已存在，检查是否接入 address 模式的搜索循环；若未接入则补全 | `keyhunt.cpp` |
| C6 | 更新 README 示例：给出 bech32 和 ETH 的 `-m address` 用法 | `README.md` |

**验证**：生成一个已知私钥的 P2WPKH 地址写入 `tests/bech32.txt`，运行 `./keyhunt -m address -f tests/bech32.txt -l compress` 应在小范围内命中。

### Track D — 测试套件

| Step | Task | Files likely to change |
|------|------|------------------------|
| D1 | 创建 `tests/generate_fixture.py`：对 8 种模式分别生成“已知私钥 -> 目标文件”的固定测试数据（范围控制在 2^20 内，秒级可跑完） | 新建 `tests/generate_fixture.py` |
| D2 | 创建 `tests/run_suite.sh`：遍历所有 fixture，调用对应 `./keyhunt` 命令，检查 stdout 是否包含 `Private Key` 或 `Key found` | 新建 `tests/run_suite.sh` |
| D3 | 在 `generate_fixture.py` 中加入负例：确保程序在无关范围内跑 5 秒不 crash | `tests/generate_fixture.py` |
| D4 | 把 `test_wif_recovery.sh` 也纳入 suite，统一输出 JUnit/TAP 风格便于查看 | `tests/test_wif_recovery.sh`, `tests/run_suite.sh` |
| D5 | 在 Makefile 添加 `make test` target，依赖 `default` 后执行 suite | `Makefile` |

**验证**：`make test` 在无 GPU 环境下全部通过；至少覆盖 address/bsgs/rmd160/eth/bech32/wif-recovery 六个模式。

---

## Integration Milestones

| Milestone | 交付物 | 前置依赖 |
|-----------|--------|----------|
| M1 | `make test` 全部通过（Track D + Track C 的 fixture） | C4, D5 |
| M2 | bech32 搜索可用，README 已更新 | C1-C6 |
| M3 | BSGS CPU 提速 ≥10x，有 `-K` 参数与 benchmark 数据 | A1-A5 |
| M4 | `make gpu` 编译通过，kernel 不再是 placeholder，有设备时自动调度 | B1-B6 |
| M5 | Release tag + CHANGELOG 更新 | M1-M4 |

---

## Risks, Tradeoffs, and Open Questions

1. **CUDA 数学库复杂度**：secp256k1 的模运算在 GPU 上极易出 bug。建议先写 `test_secp_cuda.cu` 单元测试，验证 `k*G` 与 CPU 结果一致再接入主循环。
2. **bech32 性能开销**：BCH 纠错码比 base58 重，但只在命中后的地址输出阶段执行，不影响搜索热路径。
3. **内存 vs 速度**：BSGS batching 会增大内存占用（需缓存更多中间点）。`-K` 参数让用户在内存受限机器上降级。
4. **开源合规**：项目使用 MIT license，引入外部 bech32 参考实现时注意 license 兼容（建议自己实现，逻辑简单）。
5. **Open Question**：主人是否需要 vanity 模式也支持 bech32？当前计划中仅覆盖 address/rmd160 模式的 bech32，vanity 模式可按需后续扩展。

---

## Files Likely to Change (Summary)

- `keyhunt.cpp` — 模式路由、CLI 参数、ETH/bech32 接入
- `bsgsd.cpp` — BSGS 批量 key 生成、endomorphism batching
- `bsgs_cuda.cu` / `wif_recovery_cuda.cu` — 真正 GPU kernel
- `secp256k1/SECP256K1.cpp` / `.h` — `GetHash160(P2WPKH, ...)`
- `secp256k1/IntGroup.cpp` — 批量 Int 运算优化
- `Makefile` — `gpu` target、test target、新增 `bech32.o`
- `README.md` / `CHANGELOG.md` — 文档
- 新建 `bech32/bech32.c` / `.h`
- 新建 `tests/generate_fixture.py` / `tests/run_suite.sh`
