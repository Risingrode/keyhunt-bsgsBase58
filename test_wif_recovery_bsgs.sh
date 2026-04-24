#!/bin/bash
# GPU BSGS WIF Recovery 测试脚本
# 使用方法: ./test_wif_recovery_bsgs.sh

set -e

echo "=========================================="
echo " GPU BSGS WIF Recovery 测试"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查keyhunt是否编译
if [ ! -f "./keyhunt" ]; then
    echo -e "${RED}[ERROR] keyhunt 未编译。请先运行 make gpu${NC}"
    exit 1
fi

# 生成测试用例
echo -e "${YELLOW}[1/4] 生成测试用例...${NC}"
python3 generate_bsgs_test_case.py

if [ ! -f "test_cases.txt" ]; then
    echo -e "${RED}[ERROR] 测试用例生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}[OK] 测试用例已生成${NC}"

# 运行测试
echo -e "${YELLOW}[2/4] 运行GPU BSGS WIF恢复测试...${NC}"

PASS=0
FAIL=0
TOTAL=0

while IFS='|' read -r PARTIAL_WIF EXPECTED_WIF PUBKEY COMPRESSED; do
    # 跳过注释行
    [[ "$PARTIAL_WIF" =~ ^#.* ]] && continue
    [[ -z "$PARTIAL_WIF" ]] && continue

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "--- 测试 #$TOTAL ---"
    echo "  部分WIF: $PARTIAL_WIF"
    echo "  期望WIF: $EXPECTED_WIF"
    echo "  公钥: ${PUBKEY:0:20}..."

    # 运行keyhunt
    if [ "$COMPRESSED" = "1" ]; then
        RESULT=$(timeout 120 ./keyhunt -m wif-recovery -p "$PARTIAL_WIF" -P "$PUBKEY" -g 2>&1 || true)
    else
        RESULT=$(timeout 120 ./keyhunt -m wif-recovery -p "$PARTIAL_WIF" -P "$PUBKEY" -g 2>&1 || true)
    fi

    # 检查结果
    if echo "$RESULT" | grep -q "SUCCESS"; then
        RECOVERED=$(echo "$RESULT" | grep "Recovered WIF:" | awk '{print $NF}')
        if [ "$RECOVERED" = "$EXPECTED_WIF" ]; then
            echo -e "  ${GREEN}[PASS] 恢复成功: $RECOVERED${NC}"
            PASS=$((PASS + 1))
        else
            echo -e "  ${RED}[FAIL] 恢复的WIF不匹配${NC}"
            echo "    期望: $EXPECTED_WIF"
            echo "    实际: $RECOVERED"
            FAIL=$((FAIL + 1))
        fi
    elif echo "$RESULT" | grep -q "No matching"; then
        echo -e "  ${RED}[FAIL] 未找到匹配${NC}"
        FAIL=$((FAIL + 1))
    else
        echo -e "  ${RED}[FAIL] 执行错误${NC}"
        echo "$RESULT" | tail -5
        FAIL=$((FAIL + 1))
    fi

done < test_cases.txt

# 统计结果
echo ""
echo "=========================================="
echo " 测试结果统计"
echo "=========================================="
echo -e " 总计: $TOTAL"
echo -e " ${GREEN}通过: $PASS${NC}"
echo -e " ${RED}失败: $FAIL${NC}"

if [ $FAIL -eq 0 ] && [ $TOTAL -gt 0 ]; then
    echo -e "\n${GREEN}所有测试通过!${NC}"
    exit 0
elif [ $TOTAL -eq 0 ]; then
    echo -e "\n${YELLOW}没有测试用例${NC}"
    exit 1
else
    echo -e "\n${RED}有 $FAIL 个测试失败${NC}"
    exit 1
fi
