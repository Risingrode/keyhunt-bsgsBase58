import hashlib
import ecdsa
import base58

# Private key
privkey_hex = "1184cd2cdd640ca42cfc3a091c51d549b2f016d454b2774019c2b2d2e08529fd"
privkey_bytes = bytes.fromhex(privkey_hex)

# Uncompressed WIF
extended_key = b"\x80" + privkey_bytes
checksum = hashlib.sha256(hashlib.sha256(extended_key).digest()).digest()[:4]
wif = base58.b58encode(extended_key + checksum).decode('utf-8')

# Compressed WIF
extended_key_comp = b"\x80" + privkey_bytes + b"\x01"
checksum_comp = hashlib.sha256(hashlib.sha256(extended_key_comp).digest()).digest()[:4]
wif_comp = base58.b58encode(extended_key_comp + checksum_comp).decode('utf-8')

# Public key
sk = ecdsa.SigningKey.from_string(privkey_bytes, curve=ecdsa.SECP256k1)
vk = sk.get_verifying_key()
pubkey_uncomp = b"\x04" + vk.to_string()
pubkey_comp = b"\x02" + vk.to_string()[:32] if vk.to_string()[63] % 2 == 0 else b"\x03" + vk.to_string()[:32]

print(f"Privkey: {privkey_hex}")
print(f"WIF Uncomp: {wif}")
print(f"Pubkey Uncomp: {pubkey_uncomp.hex()}")
print(f"WIF Comp: {wif_comp}")
print(f"Pubkey Comp: {pubkey_comp.hex()}")
