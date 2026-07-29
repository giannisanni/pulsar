import struct, sys, collections

path = sys.argv[1] if len(sys.argv) > 1 else "/home/models/Ornith-397B-Q2_K.gguf"
f = open(path, "rb")
f.read(4)
struct.unpack("<I", f.read(4))[0]
n_tensors = struct.unpack("<Q", f.read(8))[0]
n_kv = struct.unpack("<Q", f.read(8))[0]


def rd_str(f):
    l = struct.unpack("<Q", f.read(8))[0]
    if l > 1 << 22:
        raise Exception(f"absurd string len {l} at offset {f.tell()-8}")
    return f.read(l).decode("utf-8", "replace")


# KV value-type enum (different from tensor-type enum!)
def rd_val(f):
    t = struct.unpack("<I", f.read(4))[0]
    if t == 0: return struct.unpack("<B", f.read(1))[0]   # UINT8
    if t == 1: return struct.unpack("<b", f.read(1))[0]   # INT8
    if t == 2: return struct.unpack("<H", f.read(2))[0]   # UINT16
    if t == 3: return struct.unpack("<h", f.read(2))[0]   # INT16
    if t == 4: return struct.unpack("<I", f.read(4))[0]   # UINT32
    if t == 5: return struct.unpack("<i", f.read(4))[0]   # INT32
    if t == 6: return struct.unpack("<f", f.read(4))[0]   # FLOAT32
    if t == 7: return struct.unpack("<?", f.read(1))[0]   # BOOL
    if t == 8: return rd_str(f)                            # STRING
    if t == 9:                                             # ARRAY
        inner = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        # consume n elements of `inner` type
        items = []
        for _ in range(n):
            if inner == 0: items.append(struct.unpack("<B", f.read(1))[0])
            elif inner == 1: items.append(struct.unpack("<b", f.read(1))[0])
            elif inner == 2: items.append(struct.unpack("<H", f.read(2))[0])
            elif inner == 3: items.append(struct.unpack("<h", f.read(2))[0])
            elif inner == 4: items.append(struct.unpack("<I", f.read(4))[0])
            elif inner == 5: items.append(struct.unpack("<i", f.read(4))[0])
            elif inner == 6: items.append(struct.unpack("<f", f.read(4))[0])
            elif inner == 7: items.append(struct.unpack("<?", f.read(1))[0])
            elif inner == 8: items.append(rd_str(f))
            elif inner == 10: items.append(struct.unpack("<Q", f.read(8))[0])
            elif inner == 11: items.append(struct.unpack("<q", f.read(8))[0])
            elif inner == 12: items.append(struct.unpack("<d", f.read(8))[0])
            else: raise Exception("array inner type " + str(inner))
        return items[:8]
    if t == 10: return struct.unpack("<Q", f.read(8))[0]  # UINT64
    if t == 11: return struct.unpack("<q", f.read(8))[0]  # INT64
    if t == 12: return struct.unpack("<d", f.read(8))[0]  # FLOAT64
    raise Exception("kv val type " + str(t))


kv = {}
for _ in range(n_kv):
    k = rd_str(f)
    v = rd_val(f)
    kv[k] = v

print("=== ALL KV metadata ===")
for k in sorted(kv):
    s = f"  {k} = {str(kv[k])[:100]}"
    print(s.encode("ascii", "replace").decode("ascii"))

# tensor type enum (different from kv value-type)
T_NAMES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
 10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",
 16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",
 21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",28:"F64",30:"BF16",31:"Q4_K_S",36:"Q2_K_S"}

tensors = []
hist = collections.Counter()
per_prefix = collections.defaultdict(collections.Counter)
for _ in range(n_tensors):
    name = rd_str(f)
    n_dims = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
    ty = struct.unpack("<I", f.read(4))[0]
    off = struct.unpack("<Q", f.read(8))[0]
    tensors.append((name, dims, ty, off))
    hist[ty] += 1

print("=== tensor type histogram ===")
for ty, cnt in sorted(hist.items()):
    print(f"  ty={ty} ({T_NAMES.get(ty,'?')}): {cnt}")

print("=== blk.0 tensors ===")
n = 0
for t in tensors:
    if t[0].startswith("blk.0."):
        print(f"  {t[0]} dims={t[1]} ty={T_NAMES.get(t[2],t[2])}")
        n += 1
        if n >= 60:
            break

print("=== non-blk tensors ===")
for t in tensors:
    if not (t[0].startswith("blk.") or t[0].startswith("layers.")):
        print(f"  {t[0]} dims={t[1]} ty={T_NAMES.get(t[2],t[2])}")

# Also classify expert tensors specifically
print("=== expert tensor sample (one of each kind) ===")
seen = set()
for t in tensors:
    base = t[0]
    # normalize: replace layer N with N, expert M with M
    parts = base.split(".")
    norm = []
    for p in parts:
        if p.isdigit():
            norm.append("#")
        else:
            norm.append(p)
    key = ".".join(norm)
    if "exps" in key or "shexp" in key or "shared" in key or "ffn_" in key or "gate" in key or "down" in key or "up_" in key:
        if key not in seen:
            seen.add(key)
            print(f"  {t[0]} dims={t[1]} ty={T_NAMES.get(t[2],t[2])}  [{key}]")
