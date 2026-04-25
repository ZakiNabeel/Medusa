\# AI Context Document: Medusa OpenCL Verification Engine



\## 1. Project Overview

This project is a high-performance systems engineering implementation of the \*\*Medusa Speculative Decoding\*\* algorithm. 

The core objective is to bypass PyTorch's sequential CPU evaluation bottleneck by porting the tree-verification step to a custom OpenCL parallel-reduction kernel. The project targets consumer-grade integrated graphics rather than high-end Nvidia CUDA environments.



\## 2. Hardware Target \& Constraints

\* \*\*Device:\*\* Intel UHD Graphics 620 (Integrated GPU)

\* \*\*Compute Constraints:\*\* Max Work-Group size of 256.

\* \*\*Memory Constraints:\*\* 64 KB Local Memory (LDS) per Work-Group. Shared system memory (RAM).

\* \*\*Language/Framework:\*\* Python 3, PyOpenCL, NumPy. (Strictly hardware-agnostic OpenCL C-code).



\## 3. The Architecture (3-Stage Pipeline)

The pipeline is fully built and resides primarily in `medusa\_opencl\_verify.py`.



\* \*\*Stage 1: Dynamic Tree Builder (CPU/Python)\*\*

&#x20; \* Replaces Medusa's standard static topologies (e.g., `medusa\_choices.py`).

&#x20; \* Applies a `confidence\_threshold` to raw Medusa head logits (post-softmax).

&#x20; \* Uses Breadth-First Search (BFS) to dynamically prune low-probability branches and expand high-probability ones.

&#x20; \* Flattens the dynamic tree into 1D arrays (`candidates`, `parent\_indices`) and a 2D matrix (`node\_logits`) to pass to the GPU.

\* \*\*Stage 2: OpenCL Verification Kernel (GPU/PyOpenCL)\*\*

&#x20; \* \*\*Grid Mapping:\*\* Assigns exactly ONE Work-Group (256 threads) to ONE tree node row. 

&#x20; \* \*\*Memory Strategy:\*\* Uses Local Memory Tiling. Threads perform coalesced, strided reads from Global Memory into the 64 KB `\_\_local` memory.

&#x20; \* \*\*Computation:\*\* Executes a standard power-of-two parallel reduction tree to find the `argmax` (highest probability token) and compares it against the `draft` token. Outputs a binary `is\_match` array.

\* \*\*Stage 3: Topological Path Tracer (CPU/Python)\*\*

&#x20; \* Takes the binary `is\_match` array and the `parent\_indices` topology map.

&#x20; \* Single-pass loop to find the longest continuous path of valid nodes starting from the root.



\## 4. Performance Benchmarks

The architecture has been successfully stress-tested on synthetic dummy data (random normal float32 distributions).

\* \*\*V1 (Baseline):\*\* NumPy sequential loop.

\* \*\*V2 (Naive Kernel):\*\* Global memory reads, 1 thread per node. Result: 10x slower than CPU.

\* \*\*V3 (Local Memory Tiling):\*\* 63 nodes. Result: \*\*1.86x faster\*\* than CPU.

\* \*\*V4 (Dynamic Pipeline):\*\* 255 nodes. Result: \*\*2.73x faster\*\* than CPU. (Successfully scales under heavy load by saturating GPU cores).



\## 5. Current State \& Next Steps

\*\*Status:\*\* The inference engine and hardware acceleration logic are 100% complete and benchmarked on synthetic data.

\*\*Next Step: Full AI Integration\*\*

1\. Do NOT train new models. 

2\. Import the `transformers` library.

3\. Download a lightweight base model (e.g., `TinyLlama-1.1B`) and its corresponding pre-trained Medusa heads.

4\. Hook the real text-prompt logits into the `build\_dynamic\_tree()` function, replacing the `np.random` mock data.

5\. Generate actual English text utilizing the OpenCL hardware-verification pipeline.

