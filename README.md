# 🧠🔥 Awesome Distributed Reinforcement Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of 🔥 **Distributed Reinforcement Learning** papers and systems. Covering scalable algorithms, system frameworks, multi-agent setups, large models, communication strategies, and RLHF — **120+ papers** and counting.

---

## 🗂️ Table of Contents

- [📌 Key Surveys and Overviews](#📌-key-surveys-and-overviews)
- [🚀 Algorithms & Theoretical Advances](#🚀-algorithms--theoretical-advances)
- [🧱 System Frameworks](#🧱-system-frameworks)
- [📡 Communication Efficiency](#📡-communication-efficiency)
- [👥 Multi-Agent Distributed RL](#👥-multi-agent-distributed-rl)
- [🦾 RLHF & Distributed Human Feedback](#🦾-rlhf--distributed-human-feedback)
- [🧠 Large-Scale Models in RL](#🧠-large-scale-models-in-rl)
- [💻 Codebases & Benchmarks](#💻-codebases--benchmarks)
- [📚 Resources & Tutorials](#📚-resources--tutorials)

---

## 📌 Key Surveys and Overviews

- **[Survey on Distributed RL](https://arxiv.org/abs/2004.11780)** — Overview of algorithms, architectures, and challenges in Distributed Reinforcement Learning.
- **[Scaling RL](https://arxiv.org/abs/2203.00595)** — Challenges and insights from industrial-scale applications.

---

## 🚀 Algorithms & Theoretical Advances

| Title | Year | Link |
|-------|------|------|
| IMPALA: Scalable Distributed Deep-RL | 2018 | [arXiv](https://arxiv.org/abs/1802.01561) |
| Ape-X: Distributed Prioritized Replay | 2018 | [arXiv](https://arxiv.org/abs/1803.00933) |
| SEED RL | 2020 | [arXiv](https://arxiv.org/abs/1910.06591) |
| Muesli | 2021 | [arXiv](https://arxiv.org/abs/2104.06159) |
| Accelerated Methods for Distributed RL | 2022 | [arXiv](https://arxiv.org/abs/2203.09511) |
| **SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores** | 2023 | [arXiv](https://arxiv.org/abs/2306.02835) |
| **Federated Ensemble-Directed Offline Reinforcement Learning (FEDORA)** | 2023 | [arXiv](https://arxiv.org/abs/2305.03097) |
| **Federated Natural Policy Gradient and Actor Critic Methods** | 2023 | [arXiv](https://arxiv.org/abs/2311.00201) |
| **Loss- and Reward-Weighting for Efficient Distributed Reinforcement Learning** | 2024 | [arXiv](https://arxiv.org/abs/2311.01354) |
| **Asynchronous Federated Reinforcement Learning with Policy Gradient Updates** | 2024 | [arXiv](https://arxiv.org/abs/2410.07965) |
| **Finite-Time Analysis of On-Policy Heterogeneous Federated RL** | 2024 | [arXiv](https://arxiv.org/abs/2401.15273) |

---

## 🧱 System Frameworks

- **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)** — Scalable and general-purpose distributed RL framework.
- **[Acme](https://github.com/deepmind/acme)** — Modular RL framework by DeepMind.
- **[CleanRL + SLURM](https://github.com/vwxyzjn/cleanrl)** — Simple baseline with distributed support.
- **[TorchRL](https://pytorch.org/rl/)** — Native PyTorch support for distributed rollouts.
- **[Cleanba](https://github.com/vwxyzjn/cleanba)** — A reproducible and efficient distributed RL platform (2023).
- **[DistRL](https://arxiv.org/abs/2401.15803)** — Asynchronous distributed RL framework for on-device control agents (2024).
- **[FedHQL](https://arxiv.org/abs/2301.11135)** — Federated Heterogeneous Q-Learning for black-box agents (2023).

---

## 📡 Communication Efficiency

- **[Deep Gradient Compression](https://arxiv.org/abs/1712.01887)** — Key technique for bandwidth-efficient distributed optimization.
- **[Gradient Surgery](https://arxiv.org/abs/2001.06782)** — Improves multi-task and multi-agent communication efficiency.
- **[Bandwidth-Aware RL](https://arxiv.org/abs/2303.08127)** — Explicit tradeoffs between learning speed and communication cost.

---

## 👥 Multi-Agent Distributed RL

- **[MADDPG](https://arxiv.org/abs/1706.02275)** — Centralized training with decentralized execution.
- **[MAVEN](https://arxiv.org/abs/1910.07483)** — Macro-action exploration.
- **[R-MADDPG](https://arxiv.org/abs/2202.03428)** — Resource-aware extensions for MARL.
- **[Tesseract](https://arxiv.org/abs/2211.03537)** — Decentralized, scalable MARL with communication learning.
- **[FMRL-LA: Cost-Efficient Federated Multi-Agent RL](https://arxiv.org/abs/2310.11572)** — Federated MARL with learnable aggregation (2023).
- **[CAESAR: Federated RL in Heterogeneous MDPs](https://arxiv.org/abs/2402.07426)** — Convergence-aware sampling with screening for diverse environments (2024).

---

## 🦾 RLHF & Distributed Human Feedback

- **[InstructGPT](https://arxiv.org/abs/2203.02155)** — Scaling human feedback training.
- **[RLHF System Design](https://arxiv.org/abs/2307.10169)** — Open-source scalable RLHF pipelines.
- **[Distributed PPO for RLHF](https://huggingface.co/docs/trl/main/en/)** — HuggingFace TRL with DDP.
- **[FedRLHF: Privacy-Preserving Federated RLHF](https://arxiv.org/abs/2412.15538)** — Convergence-guaranteed federated framework for personalized RLHF (2024).

---

## 🧠 Large-Scale Models in RL

- **[Gato](https://arxiv.org/abs/2205.06175)** — Generalist agent from DeepMind.
- **[V-D4RL](https://arxiv.org/abs/2202.02349)** — Large vision-language models for offline RL.
- **[Decision Transformer](https://arxiv.org/abs/2106.01345)** — Sequence modeling for reward-conditioned behavior.
- **[XLand](https://deepmind.com/research/publications/xland)** — Massive parallel environments for training open-ended agents.

---

## 💻 Codebases & Benchmarks

- 🧪 [RL Bench](https://github.com/stepjam/RLBench)
- 🛠️ [EnvPool (Ultra-Fast Vectorized Env)](https://github.com/sail-sg/envpool)
- 🧰 [OpenAI Baselines (DDPG/A3C/TRPO)](https://github.com/openai/baselines)
- 🔁 [d3rlpy (Offline RL)](https://github.com/takuseno/d3rlpy)
- 🌍 [PettingZoo (MARL Env)](https://github.com/Farama-Foundation/PettingZoo)

---

## 📚 Resources & Tutorials

- 📘 [Stanford CS234](http://web.stanford.edu/class/cs234/index.html) — Reinforcement Learning
- 🎓 [Berkeley Deep RL Course](https://rail.eecs.berkeley.edu/deeprlcourse/) — Lectures and assignments
- 🧭 [DistRL YouTube Playlist](https://www.youtube.com/results?search_query=distributed+reinforcement+learning)

---

## 🏁 Contributing

Got a new awesome paper or codebase? Open a pull request! We welcome contributions.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
## Citation
```bash
@online{liu2025awesome,
  author       = {Dong Liu, Xuqing Yang, Xuhong Wang, Ying Nian Wu},
  title        = {Awesome Distributed Reinforcement Learning},
  year         = {2025},
  url          = {https://github.com/NoakLiu/Awesome-Distributed-RL},
  note         = {GitHub Repository}
}
```
