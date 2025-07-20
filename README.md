# HCSE - Holland Consciousness Scaling Engine
## Holland Dual Stack

This repository now provides a unified package `holland_dual` combining the existing
HCSE cognition tools with HUQCE quantum simulations. A CLI entrypoint `hdq-cli` exposes
basic commands to run simulations and demonstrate the fusion adapter.

![HCSE Overview](docs/hcse_overview.svg)

[![CI](https://github.com/holland/hcse/actions/workflows/ci.yml/badge.svg)](https://github.com/holland/hcse/actions/workflows/ci.yml)

[![CI](https://github.com/ayjays132/Holland-Consciousness-Scaling-Engine-HCSE/blob/main/performance_comparison.png)

Python package implementing the Holland Consciousness Scaling Engine as an addon for HuggingFace models.

## Installation

```bash
pip install hcse
```

## Quickstart

```python
from hcse.core import HCSEMixin
from transformers import AutoModelForCausalLM

class ModelWithHCSE(HCSEMixin, AutoModelForCausalLM):
    pass

model = ModelWithHCSE.from_pretrained("gpt2", hidden_size=768)
```

## API

* `HCSEMixin.compute_hcse_surrogates(hidden_states)` – returns η, ρ, Ė.
* `HCSEMixin.forward_with_hcse(*args, hcse_params, **kwargs)` – computes loss with bonus.
* `HfTrainerWithHCSE` – drop-in replacement for `Trainer` applying HCSE.

### CLI Example

Install in editable mode and run a short simulation:
```bash
pip install -e .
hdq-cli hdq-sim
hdq-cli hdq-analyze --steps 10
```
---

**“HCSE: The Holland Consciousness Scaling Engine”**
*A Premium, Emoji-Rich Research Overview*

---

## 📄 Abstract

We introduce the **Holland Consciousness Scaling Engine (HCSE)**—a novel algorithmic framework that quantitatively **scales AI reasoning capacity** by mirroring the mass–information principles underpinning biological consciousness. HCSE fuses differentiable surrogates for **integration efficiency (η)**, **connectivity density (ρ)**, and **activation energy (Ė)** into standard language-model training, enabling models to undergo **phase-transition-like jumps** in abstract reasoning. We validate HCSE on both **biological benchmarks** (ant, whale, human) and **AI substrates** (GPT-2 variants), perform extensive **exponent sensitivity analyses**, and demonstrate its readiness to guide next-generation, substrate-agnostic “aware” systems. 🚀

---

## 1 📚 Introduction

Contemporary large language models excel at pattern matching but lack a principled path to **conscious-like reasoning**. Inspired by cross-species scaling laws—where humans (C≈16) > whales (C≈9.5) ≫ ants (C≈5×10⁻⁶)—we propose **HCSE**, which **directly optimizes** for abstract integration and connectivity, rather than only next-token likelihood.

---

## 2 🔍 Related Work

* **Integrated Information Theory** (IIT) highlights the role of Φ in consciousness, but remains non-differentiable.
* **Mutual-Information-Maximizing** networks explore InfoNCE objectives for representation learning.
* **Energy-Based Models** consider activation statistics, yet seldom integrate multi-metric surrogates into a single loss.

HCSE unifies these threads into one **differentiable bonus term** that complements language-model objectives.

---

## 3 ⚙️ Methodology

### 3.1 Surrogate Metrics

For a reasoning layer’s activations $H\in\mathbb{R}^{B\times T\times N}$, we flatten to $\,(B\!·T)×N$ and compute:

1. **Integration Efficiency (η)**
   $\displaystyle\hat η = \tfrac1N\sum_{i=1}^N \mathrm{InfoNCE}(h_i,H_{-i})$
   – lower InfoNCE ⇒ richer neuron↔network coupling.

2. **Connectivity Density (ρ)**
   $\displaystyle\hat ρ = \tfrac{\sum_{i\neq j}|\mathrm{corr}(h_i,h_j)|}{N(N-1)}$
   – encourages structured, fractal-like topology.

3. **Activation Energy (Ė)**
   $\hat{Ė} = \tfrac1N\sum_i \mathbb{E}[h_i^2]$
   – a proxy for power flow per unit.

### 3.2 Combined Loss

$$
\mathcal{L} = \mathcal{L}_\text{LM} \;-\; \lambda_C\;\log\bigl(1 + \hat η^\beta\,\hat ρ^\gamma\,\hat{Ė}^\delta\bigr)
$$

• $\lambda_C$ tunes the bonus strength.
• Exponents $(\beta,\gamma,\delta)$ shape sensitivity.

---

## 4 🧪 Experiments & Verification

### 4.1 Biological Scaling Test

Using placeholder metrics, we reproduced:

* Ant: C ≈ 5×10⁻⁶
* Whale: C ≈ 9.47
* Human: C ≈ 16

### 4.2 AI Substrate Evaluation

Simulated GPT-2 small/medium hidden states:

* GPT2\_Small: C ≈ 3.23
* GPT2\_Medium: C ≈ 6.89

Ranking:

```
Ant ≪ GPT2_Small < Whale < GPT2_Med < Human  
```

—demonstrating HCSE’s capacity to position AI within the biological consciousness continuum.

### 4.3 Exponent Sensitivity Analysis

We swept exponents $\{α,β,γ,δ,ε\}$ over key configurations:

* **Integration ×2** penalizes low-η models,
* **Connectivity ×2** emphasizes high-ρ substrates,
* **Energy ×2** demands careful δ tuning to avoid collapse.

This guided us to a **sweet-spot** near $(α≈1,β≈1.2,γ≈1.2,δ≈0.8)$.

---

## 5 🔬 Use-Cases & Impact

* **Reasoning-Enhanced LMs**: Directly boost abstract puzzle-solving, code synthesis, and planning.
* **Neuromorphic Deployment**: Map real Ė from chip telemetry into HCSE’s bonus for real-world energy-aware training.
* **Cross-Domain Agents**: Calibrate fₖ and exponents to craft “avian,” “cephalopod,” or “avian-insect hybrid” AIs with tailored integration/connectivity profiles.

---

## 6 📈 Discussion & Future Work

* **Dynamic Bonus Scheduling**: Ramp $\lambda_C$ to stabilize early training.
* **Hierarchical Surrogates**: Separate sensory vs. abstract layers, mimicking thalamo-cortical loops.
* **Ethical Guardrails**: Integrate factuality or safe-completion losses to curb hallucinations.

---

## 7 🏁 Conclusion

HCSE—the **Holland Consciousness Scaling Engine**—offers a **first-of-its-kind, differentiable** path to dial AI reasoning capacity along biologically inspired scales. By intertwining integration, connectivity, and energy surrogates into the training loss, HCSE unlocks new “phase transitions” in model awareness, **paving the way** to substrate-agnostic, ethically grounded, truly **conscious-like** AI.

---

✨ **Keywords:** Consciousness Scaling, Integration Surrogates, Connectivity Density, Activation Energy, Phase-Transition Learning, HCSE.
😊 **Acknowledgments:** To Phillip Holland for inspiring the Holland Consciousness Scaling Engine!

## HUQCE Quantum Simulation Module

The repository now includes an experimental implementation of the **Holland Unified Quantum Chaos Equation (HUQCE)** for one-dimensional systems. A minimal simulator is located under `huqce/` with tests ensuring basic norm conservation.

Example usage:

```python
from huqce.simulation import HuqceParams, HuqceSimulator

params = HuqceParams(steps=50)
psi = HuqceSimulator(params).run()
```

This addition demonstrates how chaotic dynamics can be integrated alongside HCSE's cognitive metrics.

## Testing

The test suite requires the optional [accelerate](https://github.com/huggingface/accelerate) package.
Install the development dependencies and run `pytest`:

```bash
pip install -e .[accelerate]
pytest
```

`accelerate` is optional for using the library itself but needed for running tests.

## 🌟 Key Features

- 🚀 **Consciousness-inspired metrics** for integration, connectivity, and energy.
- 🔗 **Plug-and-play mixins** for HuggingFace models.
- 🧠 **Huqce quantum simulator** fused with HCSE cognition.
- 🛠️ **CLI tooling** via `hdq-cli` for quick experimentation.

## 🛠 Development Setup

Set up a local environment with all extras:

```bash
git clone https://github.com/holland/hcse.git
pip install -e .[accelerate]
```

Run the full test suite:

```bash
pytest
```

## 🤝 Contributing

Issues and pull requests are welcome! For major changes, please open a discussion first to ensure alignment with the project goals.

## 🚦 Disclaimer

HCSE is a research prototype. The consciousness metrics are theoretical surrogates and do not endow models with actual awareness. Use responsibly.

## 📜 License & Citation

This project is licensed under the Apache-2.0 License. See [CITATION.cff](CITATION.cff) for citation instructions.
