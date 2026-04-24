#!/usr/bin/env python3
"""
Generate WIF recovery test fixtures.

Outputs:
- test_cases.txt: partial WIF, expected WIF, public key, compression flag
- test_commands.sh: ready-to-run cache/CPU/GPU recovery commands
"""

import argparse
import hashlib
import os
import random
import shlex
from pathlib import Path


P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def modinv(a, m=P):
    if a < 0:
        a %= m
    return pow(a, -1, m)


def point_add(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        if y1 != y2:
            return None
        return point_double(p1)
    lam = ((y2 - y1) * modinv(x2 - x1, P)) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return x3, y3


def point_double(point):
    if point is None:
        return None
    x, y = point
    lam = ((3 * x * x) * modinv(2 * y, P)) % P
    x3 = (lam * lam - 2 * x) % P
    y3 = (lam * (x - x3) - y) % P
    return x3, y3


def scalar_mul(k, point=None):
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


def b58encode(payload: bytes) -> str:
    num = int.from_bytes(payload, "big")
    encoded = ""
    while num > 0:
        num, remainder = divmod(num, 58)
        encoded = BASE58_ALPHABET[remainder] + encoded
    for byte in payload:
        if byte == 0:
            encoded = "1" + encoded
        else:
            break
    return encoded


def private_key_to_wif(private_key: int, compressed=True) -> str:
    key_bytes = private_key.to_bytes(32, "big")
    payload = b"\x80" + key_bytes + (b"\x01" if compressed else b"")
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return b58encode(payload + checksum)


def private_key_to_pubkey_hex(private_key: int, compressed=True) -> str:
    x, y = scalar_mul(private_key)
    if compressed:
        prefix = "02" if y % 2 == 0 else "03"
        return prefix + format(x, "064x")
    return "04" + format(x, "064x") + format(y, "064x")


def create_partial_wif(wif: str, num_missing: int, rng: random.Random):
    if num_missing < 1 or num_missing >= len(wif):
        raise ValueError(f"invalid missing count {num_missing} for WIF length {len(wif)}")
    positions = sorted(rng.sample(range(len(wif)), num_missing))
    partial = list(wif)
    for pos in positions:
        partial[pos] = "*"
    return "".join(partial), positions


def parse_missing_counts(value: str):
    counts = []
    for item in value.split(","):
        item = item.strip()
        if item:
            counts.append(int(item, 0))
    if not counts:
        raise argparse.ArgumentTypeError("missing counts must not be empty")
    return counts


def build_keyhunt_command(partial, pubkey, n_value, threads, gpu=False):
    parts = [
        "./keyhunt",
        "-m",
        "wif-recovery",
        "-p",
        partial,
        "-P",
        pubkey,
        "-n",
        n_value,
        "-S",
        "-q",
        "-s",
        "0",
    ]
    if threads:
        parts.extend(["-t", str(threads)])
    if gpu:
        parts.append("-g")
    return " ".join(shlex.quote(part) for part in parts)


def generate_test_cases(args):
    rng = random.Random(args.seed)
    missing_counts = parse_missing_counts(args.missing)
    compressed_plan = [True] * len(missing_counts)
    if args.include_uncompressed and compressed_plan:
        compressed_plan[-1] = False

    test_cases = []
    for idx, missing_count in enumerate(missing_counts, start=1):
        compressed = compressed_plan[idx - 1]
        label = "compressed" if compressed else "uncompressed"
        print(f"Generating test case {idx}: {missing_count} missing chars ({label})")
        private_key = rng.randint(1, N - 1)
        wif = private_key_to_wif(private_key, compressed=compressed)
        pubkey = private_key_to_pubkey_hex(private_key, compressed=compressed)
        partial, positions = create_partial_wif(wif, missing_count, rng)
        test_cases.append(
            {
                "partial": partial,
                "expected": wif,
                "pubkey": pubkey,
                "compressed": int(compressed),
                "positions": positions,
            }
        )
    return test_cases


def write_outputs(test_cases, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = output_dir / "test_cases.txt"
    commands_path = output_dir / "test_commands.sh"

    with cases_path.open("w", encoding="utf-8") as f:
        f.write("# partial_wif|expected_wif|pubkey|compressed\n")
        for case in test_cases:
            f.write(
                f"{case['partial']}|{case['expected']}|{case['pubkey']}|{case['compressed']}\n"
            )

    with commands_path.open("w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n\n")
        f.write("# These commands intentionally include -S so the BSGS cache files are read or generated first.\n\n")
        for idx, case in enumerate(test_cases, start=1):
            cpu_cmd = build_keyhunt_command(
                case["partial"],
                case["pubkey"],
                args.n,
                args.threads,
                gpu=False,
            )
            gpu_cmd = build_keyhunt_command(
                case["partial"],
                case["pubkey"],
                args.n,
                args.threads,
                gpu=True,
            )
            f.write(f"echo '=== Test case {idx}: CPU BSGS/cache path ==='\n")
            f.write(cpu_cmd + "\n\n")
            f.write(f"echo '=== Test case {idx}: CUDA WIF recovery path ==='\n")
            f.write(gpu_cmd + "\n\n")

    os.chmod(commands_path, 0o755)
    return cases_path, commands_path


def main():
    parser = argparse.ArgumentParser(description="Generate WIF/public-key BSGS recovery fixtures.")
    parser.add_argument("--output-dir", default=".", help="directory for test_cases.txt and test_commands.sh")
    parser.add_argument("--seed", type=int, default=None, help="deterministic random seed")
    parser.add_argument("--missing", default="1,2,3,2", help="comma-separated wildcard counts")
    parser.add_argument("--n", default="0x100000", help="BSGS -n value to use in generated commands")
    parser.add_argument("--threads", type=int, default=0, help="optional -t value in generated commands")
    parser.add_argument(
        "--no-uncompressed",
        dest="include_uncompressed",
        action="store_false",
        help="do not make the last fixture uncompressed",
    )
    parser.set_defaults(include_uncompressed=True)
    args = parser.parse_args()

    test_cases = generate_test_cases(args)
    cases_path, commands_path = write_outputs(test_cases, args)

    print(f"Generated {len(test_cases)} test cases -> {cases_path}")
    print(f"Generated runnable commands -> {commands_path}")
    for idx, case in enumerate(test_cases, start=1):
        print(f"\nTest #{idx}:")
        print(f"  Partial WIF: {case['partial']}")
        print(f"  Expected WIF: {case['expected']}")
        print(f"  Public key: {case['pubkey']}")
        print(f"  Missing positions: {','.join(str(p) for p in case['positions'])}")
        print(f"  Compressed: {'yes' if case['compressed'] else 'no'}")
        print("  CPU command:")
        print("   ", build_keyhunt_command(case["partial"], case["pubkey"], args.n, args.threads))
        print("  GPU command:")
        print("   ", build_keyhunt_command(case["partial"], case["pubkey"], args.n, args.threads, gpu=True))


if __name__ == "__main__":
    main()
