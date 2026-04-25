from tensor import MemoryPool, Tensor

print("🚀 Starting Engine...")

try:
    # 1. Create Pool
    pool = MemoryPool(capacity_bytes=1024 * 1024)
    
    # 2. Test Math (This will cross the C-Bridge into the Engine)
    A = Tensor.ones(pool, shape=(2, 2))
    B = Tensor.ones(pool, shape=(2, 2))
    
    C = A + B
    
    print("✅ Connection Succeeded! Output:")
    print(C.numpy())

except Exception as e:
    print("❌ Connection Failed:")
    print(e)