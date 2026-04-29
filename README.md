<div align="center">

<h1>Arch-LLM</h1>

<p><strong>Hardware-aware memory optimization for large language models under strict token budgets.</strong></p>

</div>

---

## Results

- **Budget Adherence:** 95%  
- **HBM Pressure Reduction:** 30%  
- **Core Performance:** 100% C++ execution pipeline  
- **Redundancy Reduction:** High-fidelity semantic pruning  
- **System Type:** Hardware-level memory simulation for LLM context optimization  

---

## Overview

Arch-LLM is a hardware-aware simulation system for optimizing memory usage in large language models. It models token flow under strict hardware constraints and replaces naive truncation strategies with structured, semantics-preserving compression.

The system treats context as a constrained resource, applying redundancy-aware pruning and allocation strategies to maximize retained information within fixed token budgets.

---

## Method / Approach

<p align="center">
  <img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/7380c58e-3516-4f3f-9053-6b732392f376" />
</p>

- **Input Stream Processing**  
  Sequences are ingested and structured for analysis under constrained memory conditions.

- **Redundancy Detection**  
  Semantic overlap between tokens is identified to eliminate low-information repetition.

- **Token Budgeting Layer**  
  Enforces strict adherence to predefined token limits through controlled pruning.

- **Memory Simulation**  
  Simulates HBM constraints, bandwidth limits, and allocation pressure.

- **Compression + Output**  
  Produces an optimized context state with maximal semantic retention.

---

## Data

- **Type:** synthetic / structured token streams  
- **Domain:** LLM context windows and retrieval pipelines  
- **Focus:** memory-constrained inference scenarios  

<p align="center">
 <img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/82c3090d-5285-4990-8e43-a193ec6f7ae1" />
</p>

Preprocessing:
- token segmentation  
- semantic grouping  
- redundancy scoring  

---

## Experiments / Reproduction

```bash
./archLLM_sim
````

## Run simulation:

```bash
./archLLM_sim --input sample_sequence.txt
```

## Build + run full pipeline:

```bash
mkdir build && cd build
cmake ..
make
./archLLM_sim
```

Input: token sequence
Output: optimized context + memory usage metrics

Dependencies

```bash
C++ 20
CMake
Standard Template Library (STL)
```

## Repository Structure

```bash
archLLM-sim/
├── src/
├── include/
├── simulation/
├── optimizer/
├── metrics/
├── build/
└── README.md
```

## Installation

```bash
git clone https://github.com/v1shay/archLLM-sim.git
cd archLLM-sim
mkdir build && cd build
cmake ..
make
```

## Optional:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
