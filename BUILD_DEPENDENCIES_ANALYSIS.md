# Build Dependencies Analysis

## 🔍 Current Build Dependencies in Dockerfile

```dockerfile
# Stage 1: Builder
RUN apt-get install -y --no-install-recommends \
    libclang-dev \
    gcc \
    g++
```

---

## 📊 Dependency Analysis

### 1. **libclang-dev** - ✅ REQUIRED

**Why it's needed:**
- Your `symbol_extractor.py` uses the Python `libclang` package
- Parses C/C++ code to extract symbols (functions, classes, structs, etc.)
- Used by: `from clang import cindex`

**What it does:**
```python
# symbol_extractor.py line 33
index = cindex.Index.create()
tu = index.parse(filepath, options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)
```

**Build vs Runtime:**
- **Build time**: Needs `libclang-dev` (headers + libraries) to install Python package
- **Runtime**: Only needs `libclang1` (shared library, no headers)

**Size impact:**
- `libclang-dev`: ~40 MB
- `libclang1`: ~15 MB
- **Savings**: ~25 MB by using runtime-only in final image ✅ (Already optimized)

---

### 2. **gcc** - ⚠️ MAYBE REQUIRED

**Potentially needed for:**
1. **faiss-cpu** - If no pre-built wheel available
2. **Some torch dependencies** - If building from source
3. **sentence-transformers dependencies** - Rare, usually has wheels

**How to test if needed:**
```bash
# Try building without gcc
docker build --target builder -t test-no-gcc -f- . <<EOF
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends libclang-dev g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EOF
```

**Expected result:**
- ✅ If successful: gcc not needed (all packages have wheels)
- ❌ If fails: gcc is required

**Size impact:**
- `gcc`: ~30 MB
- **Potential savings**: ~30 MB if not needed

---

### 3. **g++** - ⚠️ LIKELY REQUIRED

**Needed for:**
1. **faiss-cpu** - Facebook's similarity search library (C++ codebase)
   - Used in `vector_store.py` for semantic code search
   - Likely needs C++ compiler if no wheel for your platform

**How it's used:**
```python
# vector_store.py
import faiss  # This is a C++ library with Python bindings
```

**Size impact:**
- `g++`: ~35 MB
- **Potential savings**: ~35 MB if faiss-cpu has pre-built wheel

---

## 🎯 Optimization Strategy

### Option A: Try Minimal Build (Recommended to test first)

```dockerfile
# Minimal - only what's absolutely needed
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
```

**Expected outcome:**
- ✅ **Success**: All packages have pre-built wheels → Save ~65 MB
- ❌ **Failure**: Some packages need compilation → Add back gcc/g++

---

### Option B: Keep g++ only (Middle ground)

```dockerfile
# If faiss-cpu needs compilation but other packages don't
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

**Savings**: ~30 MB (removes gcc)

---

### Option C: Current Setup (Safe but larger)

```dockerfile
# Current - includes all compilers
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    libclang-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
```

**Savings**: 0 MB (but guaranteed to work)

---

## 🧪 Testing Plan

### Step 1: Test minimal build
```bash
# Backup current Dockerfile
cp Dockerfile Dockerfile.backup

# Try minimal build (Option A)
# Edit Dockerfile to remove gcc and g++
docker-compose build --no-cache

# If successful:
echo "✅ gcc/g++ not needed - saved ~65 MB"

# If fails with compilation errors:
echo "❌ Need to add back compilers"
```

### Step 2: Check which package failed
```bash
# Look for error messages like:
# "error: command 'gcc' failed"
# "error: command 'g++' failed"
# "building 'faiss._swigfaiss' extension"
```

### Step 3: Add back only what's needed
```bash
# If faiss-cpu failed → add g++
# If other packages failed → add gcc
# If both failed → keep current setup
```

---

## 📈 Expected Savings Summary

| Scenario | Compilers Removed | Size Savings | Likelihood |
|----------|------------------|--------------|------------|
| Best case | gcc + g++ | ~65 MB | 30% (all wheels available) |
| Middle case | gcc only | ~30 MB | 50% (faiss needs g++) |
| Worst case | None | 0 MB | 20% (platform has no wheels) |

---

## 🎓 Why Pre-built Wheels Matter

**Pre-built wheels** are pre-compiled Python packages:
- ✅ No compilation needed during `pip install`
- ✅ No gcc/g++ required
- ✅ Faster installation
- ✅ Smaller Docker images

**When wheels are NOT available:**
- Platform is uncommon (e.g., ARM, Alpine Linux)
- Package is very new or niche
- Package has complex C/C++ dependencies

**Your platform** (python:3.12-slim = Debian on x86_64):
- ✅ Very common platform
- ✅ Most packages have wheels
- ✅ Good chance gcc/g++ not needed

---

## 💡 Recommendation

**Try this order:**

1. **First**: Remove gcc and g++, test build
   - If works → **Save ~65 MB** ✅
   
2. **If fails**: Add back g++ only, test build
   - If works → **Save ~30 MB** ✅
   
3. **If still fails**: Add back gcc too
   - Keep current setup → **Save 0 MB** but guaranteed to work

**My prediction**: You'll likely need g++ for faiss-cpu, but not gcc.
**Expected savings**: ~30 MB

---

## 🔧 Quick Test Command

```bash
# Test if your packages need gcc/g++
docker run --rm python:3.12-slim bash -c "
  apt-get update && 
  apt-get install -y libclang-dev && 
  pip install --no-cache-dir torch>=2.0.0 sentence-transformers faiss-cpu libclang
"

# If this succeeds, you don't need gcc/g++!
# If it fails, check the error message to see which compiler is needed
```

