#!/usr/bin/env python3
"""
生成GPU BSGS WIF恢复测试用例
- 生成随机私钥
- 转换为WIF
- 创建部分WIF（隐藏若干字符）
- 计算对应的公钥
- 输出测试用例
"""

import hashlib
import os
import sys

# secp256k1 参数
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def modinv(a, m=P):
    """扩展欧几里得算法求模逆"""
    if a < 0:
        a = a % m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception('模逆不存在')
    return x % m

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x

def point_add(p1, p2):
    """椭圆曲线点加法"""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if y1 != y2:
            return None
        else:
            return point_double(p1)
    lam = ((y2 - y1) * modinv(x2 - x1, P)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)

def point_double(p):
    """椭圆曲线点倍乘"""
    if p is None:
        return None
    x, y = p
    lam = ((3 * x * x) * modinv(2 * y, P)) % P
    x3 = (lam * lam - 2 * x) % P
    y3 = (lam * (x - x3) - y) % P
    return (x3, y3)

def scalar_mul(k, point=None):
    """标量乘法 k*G"""
    if point is None:
        point = (Gx, Gy)
    result = None
    addend = point
    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1
    return result

def private_key_to_wif(private_key, compressed=True):
    """私钥转WIF"""
    # 32字节大端
    key_bytes = private_key.to_bytes(32, 'big')

    if compressed:
        extended = b'\x80' + key_bytes + b'\x01'
    else:
        extended = b'\x80' + key_bytes

    # 双SHA256校验和
    first_sha = hashlib.sha256(extended).digest()
    second_sha = hashlib.sha256(first_sha).digest()
    checksum = second_sha[:4]

    payload = extended + checksum

    # Base58编码
    num = int.from_bytes(payload, 'big')
    encoded = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = BASE58_ALPHABET[remainder] + encoded

    # 处理前导零字节
    for byte in payload:
        if byte == 0:
            encoded = '1' + encoded
        else:
            break

    return encoded

def private_key_to_compressed_pubkey_hex(private_key):
    """私钥转压缩公钥hex"""
    point = scalar_mul(private_key)
    x, y = point
    prefix = '02' if y % 2 == 0 else '03'
    return prefix + format(x, '064x')

def private_key_to_uncompressed_pubkey_hex(private_key):
    """私钥转非压缩公钥hex"""
    point = scalar_mul(private_key)
    x, y = point
    return '04' + format(x, '064x') + format(y, '064x')

def create_partial_wif(wif, num_missing):
    """创建部分WIF，隐藏指定数量的字符"""
    import random
    positions = random.sample(range(len(wif)), num_missing)
    partial = list(wif)
    for pos in positions:
        partial[pos] = '*'
    return ''.join(partial), positions

def generate_test_cases():
    """生成测试用例"""
    import random

    test_cases = []

    # 测试用例1: 3个缺失字符 (简单)
    print("生成测试用例1: 3个缺失字符...")
    privkey1 = random.randint(1, N-1)
    wif1 = private_key_to_wif(privkey1, compressed=True)
    pubkey1 = private_key_to_compressed_pubkey_hex(privkey1)
    partial1, _ = create_partial_wif(wif1, 3)
    test_cases.append((partial1, wif1, pubkey1, 1))

    # 测试用例2: 5个缺失字符 (中等)
    print("生成测试用例2: 5个缺失字符...")
    privkey2 = random.randint(1, N-1)
    wif2 = private_key_to_wif(privkey2, compressed=True)
    pubkey2 = private_key_to_compressed_pubkey_hex(privkey2)
    partial2, _ = create_partial_wif(wif2, 5)
    test_cases.append((partial2, wif2, pubkey2, 1))

    # 测试用例3: 7个缺失字符 (较难)
    print("生成测试用例3: 7个缺失字符...")
    privkey3 = random.randint(1, N-1)
    wif3 = private_key_to_wif(privkey3, compressed=True)
    pubkey3 = private_key_to_compressed_pubkey_hex(privkey3)
    partial3, _ = create_partial_wif(wif3, 7)
    test_cases.append((partial3, wif3, pubkey3, 1))

    # 测试用例4: 4个缺失字符 (非压缩)
    print("生成测试用例4: 4个缺失字符 (非压缩)...")
    privkey4 = random.randint(1, N-1)
    wif4 = private_key_to_wif(privkey4, compressed=False)
    pubkey4 = private_key_to_uncompressed_pubkey_hex(privkey4)
    partial4, _ = create_partial_wif(wif4, 4)
    test_cases.append((partial4, wif4, pubkey4, 0))

    # 写入文件
    with open('test_cases.txt', 'w') as f:
        f.write("# 部分WIF | 期望WIF | 公钥 | 是否压缩\n")
        for partial, expected, pubkey, compressed in test_cases:
            f.write(f"{partial}|{expected}|{pubkey}|{compressed}\n")

    print(f"已生成 {len(test_cases)} 个测试用例 → test_cases.txt")

    # 同时输出到屏幕
    for i, (partial, expected, pubkey, compressed) in enumerate(test_cases):
        print(f"\n测试 #{i+1}:")
        print(f"  部分WIF: {partial}")
        print(f"  期望WIF: {expected}")
        print(f"  公钥: {pubkey[:40]}...")
        print(f"  压缩: {'是' if compressed else '否'}")

if __name__ == '__main__':
    generate_test_cases()
