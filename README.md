<div align="center">

<h1>Arch-LLM</h1>

<p><strong>ArchLLM is a memory-aware inference architecture designed to power the Arch1 model</strong></p>

</div>

---

## Results

- STATE-Bench Accuracy: **92.4%**
- Budget Adherence: **95%**
- HBM Pressure Reduction: **30%**
- Core Performance: **CUDA-accelerated execution pipeline**

---

## Overview

<div align="center">

<img width="971" height="647" alt="Screenshot 2026-07-07 at 4 03 45 PM" src="https://github.com/user-attachments/assets/aec0c22b-3d48-4997-ad90-98aa37865cd7" />

<p><em>Arch1 layers the ArchLLM harness over Qwen3.6, written with CUDA for accelerated execution on NVIDIA GPUs and orchestrated through Kubernetes for scalable inference</em></p>

</div>

---

## Evaluation

<div align="center">

<img width="1178" height="626" alt="Screenshot 2026-07-07 at 4 00 59 PM" src="https://github.com/user-attachments/assets/1873c73d-d212-4886-84b0-985637c03d68" />

<p><em>Synthetic STATE-Bench evaluation demonstrates the performance of Arch1 under the ArchLLM inference architecture</em></p>

</div>

---

## Method / Approach

<p align="center">
  <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7380c58e-3516-4f3f-9053-6b732392f376" />
</p>

<p align="center"><em>The ArchLLM optimization pipeline minimizes redundant context movement while preserving high-value memory for long-context inference</em></p>

---

## Data

<p align="center">
  <img width="1536" height="1024" alt="Screenshot 2026-07-07 at 4 15 30 PM" src="https://github.com/user-attachments/assets/ef850de3-2740-4f31-89d9-21bf0fc696e6" />
</p>

<p align="center"><em>The evaluation pipeline transforms token streams into optimized memory layouts and GPU-level efficiency metrics</em></p>

---

## Experiments / Reproduction

```bash
./archLLM_sim
```

### Run simulation

```bash
./archLLM_sim --input sample_sequence.txt
```

### Build + run full pipeline

```bash
mkdir build && cd build
cmake ..
make
./archLLM_sim
```

**Input:** Token sequence

**Output:** Optimized context and memory usage metrics

---

## Dependencies

```text
CUDA Toolkit
CMake
C++20
Standard Template Library (STL)
```

---

## Installation

```bash
git clone https://github.com/v1shay/archLLM-sim.git
cd archLLM-sim
mkdir build && cd build
cmake ..
make
```

### Optional

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
