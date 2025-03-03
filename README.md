<div align="center">
  <img src="src/repo/image/logo.png" alt="Logo-Banner" width="80%" max-width="485px">
</div>

# Reno-Vans
> __"Quaestio Tua, Scientiam Renovans."__

###
<div align="center">
  <img src="src/repo/svg/bar.svg" alt="Animated Banner" style="width:100%; max-width:1280px; height:auto; margin:20px auto; display:block; padding-top:20px; padding-bottom:20px;">
</div>

###

<!--
style path: src/repo/svg/bar.svg

In case of update:

Available:
![Available](https://img.shields.io/badge/model-available-success)

Unavailable:
![Unavailable](https://img.shields.io/badge/model-unavailable-red)

In Development:
![In Development](https://img.shields.io/badge/model-in%20development-yellow)
-->

### Models
1. __Reno-Orion Nous__ ![In Development](https://img.shields.io/badge/model-in%20development-yellow)
   - Naming Convention: Orion (strength/versatility) + Nous (intellect/understanding).
   - Description:
     - Our flagship general-purpose AI. Reno-Orion Nous combines raw processing power with deep cognitive abilities. It excels in complex reasoning, problem-solving, and handling a wide range of tasks. Think of it as a powerful and deeply intelligent partner.

2. __Reno-Aquila Ingenium__ ![Unavailable](https://img.shields.io/badge/model-unavailable-red)
   - Naming Convention: Aquila (vision/swiftness) + Ingenium (ingenuity/resourcefulness).
   - Description:
     - Optimized for analytical tasks, data processing, and rapid problem-solving. Reno-Aquila Ingenium is highly efficient and capable of finding innovative solutions. Imagine a very fast and creative problem solver, adept at extracting insights and generating novel approaches.

3. __Reno-Lyra Veritas__ ![Unavailable](https://img.shields.io/badge/model-unavailable-red)
   - Naming Convention: Lyra (harmony/creativity) + Veritas (truth)
   - Description:
     - Designed for creative tasks that require accuracy and factual grounding. Reno-Lyra Veritas excels in producing outputs that are both beautiful and correct. Consider it a creative writer or artist with the precision of a fact-checker

###
<div align="center">
  <img src="src/repo/svg/bar.svg" alt="Animated Banner" style="width:100%; max-width:1280px; height:auto; margin:20px auto; display:block; padding-top:20px; padding-bottom:20px;">
</div>

###

<div align="center">
  
  <!-- Project Status & Version -->
  ![Status](https://img.shields.io/badge/Status-In_Development-yellow)
  ![Version](https://img.shields.io/badge/Version-0.1.0-blue)
  ![License](https://img.shields.io/badge/License-CC_BY--NC--ND-lightgrey)
  
  <!-- Core Models -->
  ![StarCoder2](https://img.shields.io/badge/StarCoder2--3B-Primary_Model-success)
  ![CodeGemma](https://img.shields.io/badge/CodeGemma--2B-Secondary_Model-success)
  ![Phi-2](https://img.shields.io/badge/Phi--2-Evaluator_Model-blueviolet)
  ![MiniLM](https://img.shields.io/badge/MiniLM--L12-Embedding_Model-blue)
  
  <!-- Core Technologies -->
  ![RLAIF](https://img.shields.io/badge/Learning-RLAIF-blueviolet)
  ![Ensemble](https://img.shields.io/badge/Architecture-Ensemble_Learning-blue)
  ![Quantization](https://img.shields.io/badge/Optimization-Mixed_Precision-green)
  
  <!-- Main Features -->
  ![Code Generation](https://img.shields.io/badge/Feature-Code_Generation-orange)
  ![Web Development](https://img.shields.io/badge/Domain-Web_Development-orange)
  ![Multi-metric Evaluation](https://img.shields.io/badge/Evaluation-Multi_metric-teal)
  
  <!-- Technical Implementation -->
  ![Logistic Regression](https://img.shields.io/badge/Algorithm-Logistic_Regression-informational)
  ![KNN](https://img.shields.io/badge/Algorithm-KNN_Similarity-informational)
  ![TF-IDF](https://img.shields.io/badge/Feature_Engineering-TF_IDF-blue)
  ![FAISS](https://img.shields.io/badge/Indexing-FAISS-blue)
  
  <!-- Optimization -->
  ![FlashAttention](https://img.shields.io/badge/Optimization-FlashAttention2-green)
  ![xFormers](https://img.shields.io/badge/Optimization-xFormers-green)
  ![KV-Cache](https://img.shields.io/badge/Optimization-KV_Cache-green)
  
  <!-- Tech Stack -->
  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
  ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFAC00?style=for-the-badge&logo=huggingface&logoColor=black)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  
  <!-- Social -->
  ![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)
</div>

###
<div align="center">
  <img src="src/repo/svg/bar.svg" alt="Animated Banner" style="width:100%; max-width:1280px; height:auto; margin:20px auto; display:block; padding-top:20px; padding-bottom:20px;">
</div>

###

## Overview

Reno‑Vans is an advanced AI ensemble system designed for automated webpage creation. By integrating multiple pre‑trained models through a robust, iterative refinement process, Reno‑Vans produces high‑quality, semantically accurate webpages with exceptional UI/UX. The system eschews traditional datasets by synthesizing its own training data and continuously improves its outputs through Reinforcement Learning from AI Feedback (RLAIF) and optional Human Feedback (RLHF). Recent updates (RE‑2) have optimized the system for maximum speed and resource efficiency, incorporating state‑of‑the‑art quantization, attention caching, and pipeline optimizations.

---

## Hardware Target

- **Processor:** Intel Core i5‑12450H  
- **GPU:** GeForce RTX 3060 4GB  
- **Storage:** 2TB SSD  

---

## Model Architecture

### Primary Model  
- **bigcode/StarCoder2-3b**  
  - The core code generator trained extensively on web development repositories.
  - Operates in 4‑bit quantization (via GPTQ) for optimal speed and memory efficiency.
  - Excels in generating HTML, CSS, and JavaScript with semantic awareness and robust infilling capabilities.

### Secondary Model  
- **google/codegemma-2b**  
  - A lightweight, fast model for inter‑model evaluation and candidate refinement.
  - Runs in 4‑bit quantization (via bitsandbytes/GGUF) to minimize overhead.
  - Provides rapid feedback to drive the deep thinking cycle.

### Evaluator (RLAIF)  
- **microsoft/Phi‑2**  
  - Serves as the RLAIF evaluator providing detailed, structured feedback.
  - Utilizes 8‑bit quantization to reduce memory consumption while maintaining accuracy.
  - Scores outputs based on correctness, design principles, execution, and static analysis.

### Reinforcement (IR‑NLP)  
- **sentence‑transformers/all‑MiniLM‑L12‑v2**  
  - Generates lightweight, high‑quality FP16 embeddings.
  - Supports fast reference retrieval via a FAISS‑backed index and aggressive caching.
  - Enhances reinforcement learning by precomputing embeddings for IR tasks.

---

## Learning Paradigms

### 1. Reinforcement Learning from AI Feedback (RLAIF)
- **Strategy:**  
  - The system employs an AI‑generated self‑critique loop where models assess each other’s outputs.
  - **microsoft/Phi‑2** provides detailed feedback which is used to rank alternatives.
  - Reward signals are generated based on simulated execution, static analysis, and advanced metrics.
  - Gradient updates incorporate feedback into the model weights continuously.

- **Architecture:**  
  - **Primary:** bigcode/StarCoder2‑3b  
  - **Secondary:** google/codegemma‑2b  
  - **Evaluator:** microsoft/Phi‑2  
  - Automatic debugging is integrated via language‑model powered DebuggerAgents.

### 2. Reinforcement Learning from Human Feedback (RLHF) (Optional)
- **Strategy:**  
  - Provides an optional interface for collecting human feedback.
  - A preference model is built to adjust generation parameters dynamically.
- **Framework:**  
  - Uses standardized evaluation criteria and long‑term feedback tracking.

### 3. IR‑NLP Enhancements
- **Strategy:**  
  - Lightweight embeddings from **sentence‑transformers/all‑MiniLM‑L12‑v2** are precomputed and stored using FAISS.
  - Aggressive caching and approximate retrieval methods optimize IR performance.

---

## Pipeline and Software Optimizations

- **Optimized Inference:**  
  - Integration with accelerated inference frameworks such as **vLLM/llama.cpp** and **ExLlama‑v2**.
- **Attention & KV‑cache:**  
  - Leverages **FlashAttention‑2**, **xFormers**, and **PagedAttention** for efficient attention computations and reduced memory usage.
- **Memory Management:**  
  - Implements CPU‑GPU offloading, gradient checkpointing, and tensor parallelism.
  - Sequential model loading/unloading and continuous batching ensure efficient VRAM usage.
  - Disk‑based checkpointing aids in rapid recovery.
- **IR‑NLP Caching:**  
  - Uses a FAISS‑based index for fast similarity search and aggressive caching of precomputed embeddings.

---

## Ensemble Integration

### Logistic Regression Ensemble
A dedicated logistic regression layer combines features from:
- **Primary confidence:** Derived from token‑ and sequence‑level entropy.
- **Secondary confidence:** Measured via ensemble disagreement between primary and secondary outputs.
- **Evaluator confidence:** Computed from evaluator model entropy.
- **Raw entropy values, disagreement scores, KNN similarity,** and **example quality metadata**.
  
This ensemble layer produces a final score that guides iterative refinement and training.

### Deep Thinking Cycle
The system generates multiple candidate refinements using the secondary model. Candidates are scored via the ensemble logistic regression layer, and the best candidate is selected based on highest aggregated score.

### Cross‑Model Alignment
Projection layers map hidden states from each model into a common space. An adversarial perturbation strategy is applied to optimize cosine similarity, ensuring consistent knowledge transfer across models.

### Automatic Debugging
A dedicated DebuggerAgent automatically generates detailed debugging reports when evaluation scores are low, with caching to avoid redundant computations.

### Advanced Knowledge Distillation
A student MLP is trained to mimic the ensemble output, distilling the knowledge of the entire system into a compact model.

---

## Evaluation Metrics

1. **Code Execution Success Rate:**  
   - Percentage of generated code that executes without errors across multiple browsers.

2. **Code Quality Metrics:**  
   - Adherence to coding standards, complexity, maintainability, and security vulnerability assessments.

3. **Functional Correctness:**  
   - Accuracy of implementation against specifications and edge case handling.

4. **UI/UX User Satisfaction:**  
   - Measured via standardized usability tests and subjective ratings.

5. **Accessibility Metrics:**  
   - 100% compliance with WCAG 2.1 AA standards.

6. **Layout and Design Quality:**  
   - Evaluation of visual hierarchy, typography, color harmony, and responsive behavior.

7. **Ensemble Consistency:**  
   - Coherence between design intent and code implementation.

8. **Overall Task Performance:**  
   - End-to-end solution completeness, integration effectiveness, and resource efficiency.

9. **Error Rate:**  
   - Frequency and severity of system failures.

10. **Prompt Engineering Evaluation:**  
    - Ability to interpret and follow varied instructions accurately.

11. **RLAIF Effectiveness:**  
    - Quality improvements from AI feedback.

12. **RLHF Alignment (Optional):**  
    - Correlation between system behavior and human preferences.

---

## Risk Management & Continuous Improvement

- **Automated Testing:**  
  - Static analysis, dynamic testing, UI regression, and load testing pipelines.
- **Error Targets:**  
  - Code execution success >98%; ensemble error rate <3% critical errors.
- **Performance:**  
  - Full webpage generation in <12 seconds, memory usage <10GB, API latency <120ms.
- **Continuous Learning:**  
  - Iterative refinement, cross‑model synchronization, and regular knowledge distillation.

---

## Let Us Know!

For reporting issues or providing feedback, please contact us via [Email](mailto:relay.arbiter303@gmail.com) or open an issue on [GitHub](https://github.com/Protostarship/reno-vans/issues).

<div align="center">
  <img src="src/repo/svg/bar.svg" alt="Animated Banner" style="width:100%; max-width:1280px; height:auto; margin:20px auto; padding:20px 0;">
</div>

---

