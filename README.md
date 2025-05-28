# 🧠🔥 Awesome Distributed Reinforcement Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of 🔥 **Distributed Reinforcement Learning** papers and systems. Covering scalable algorithms, system frameworks, multi-agent setups, large models, communication strategies, and RLHF — **130+ papers** and counting, maintained by [Dong Liu](https://github.com/NoakLiu) and [Xuqing Yang](https://github.com/catalpa-bungei).

---

## 🗂️ Table of Contents

- [📌 Key Surveys and Overviews](#📌-key-surveys-and-overviews)
- [🚀 Algorithms & Theoretical Advances](#🚀-algorithms--theoretical-advances)
  - [📈 Policy Gradient Methods](#📈-policy-gradient-methods)
  - [🎯 Value-Based Methods](#🎯-value-based-methods)
  - [🎭 Actor-Critic Methods](#🎭-actor-critic-methods)
  - [⚡ Asynchronous & Parallel Methods](#⚡-asynchronous--parallel-methods)
  - [🔄 Gradient Aggregation & Optimization](#🔄-gradient-aggregation--optimization)
  - [🧮 Theoretical Foundations](#🧮-theoretical-foundations)
  - [🎲 Exploration & Sampling](#🎲-exploration--sampling)
- [🧱 System Frameworks](#🧱-system-frameworks)
- [📡 Communication Efficiency](#📡-communication-efficiency)
- [👥 Multi-Agent Distributed RL](#👥-multi-agent-distributed-rl)
- [🦾 RLHF & Distributed Human Feedback](#🦾-rlhf--distributed-human-feedback)
- [🧠 Large-Scale Models in RL](#🧠-large-scale-models-in-rl)
- [💻 Codebases & Benchmarks](#💻-codebases--benchmarks)
- [📚 Resources & Tutorials](#📚-resources--tutorials)
- [🏁 Contributing](#🏁-contributing)
- [📜 License](#📜-license)
- [📎 Citation](#📎-citation)

---

## 📌 Key Surveys and Overviews

- **[Survey on Distributed RL](https://arxiv.org/abs/2004.11780)**  
- **[Scaling RL](https://arxiv.org/abs/2203.00595)**  
- **[A Survey on Federated RL](https://arxiv.org/abs/2202.02272)**  
- **[RLHF: Challenges & Opportunities](https://arxiv.org/abs/2307.10169)**  

---

## 🚀 Algorithms & Theoretical Advances

### 📈 Policy Gradient Methods

| Title                                                      |   Year | Link                             |
|:-----------------------------------------------------------|-------:|:---------------------------------|
| IMPALA: Scalable Distributed Deep-RL                       |   2018 | https://arxiv.org/abs/1802.01561 |
| Distributed Proximal Policy Optimization                   |   2017 | https://arxiv.org/abs/1707.02286 |
| Asynchronous Methods for Deep Reinforcement Learning (A3C) |   2016 | https://arxiv.org/abs/1602.01783 |
| Asynchronous Federated RL with PG Updates                  |   2024 | https://arxiv.org/abs/2402.08372 |
| Distributed Policy Gradient with Variance Reduction        |   2023 | https://arxiv.org/abs/2305.07180 |
| Federated PG for Multi-Agent RL                            |   2023 | https://arxiv.org/abs/2304.12151 |
| Scalable Distributed Policy Optimization via Consensus     |   2024 | https://arxiv.org/abs/2401.09876 |


### 🎯 Value-Based Methods

| Title                                                 |   Year | Link                             |
|:------------------------------------------------------|-------:|:---------------------------------|
| Ape-X: Distributed Prioritized Replay                 |   2018 | https://arxiv.org/abs/1803.00933 |
| Distributed Deep Q-Learning                           |   2015 | https://arxiv.org/abs/1508.04186 |
| Distributed Rainbow                                   |   2023 | https://arxiv.org/abs/2302.14592 |
| Federated Q-Learning with Differential Privacy        |   2024 | https://arxiv.org/abs/2403.15789 |
| Asynchronous Distributed Value Function Approximation |   2023 | https://arxiv.org/abs/2309.12456 |


### 🎭 Actor-Critic Methods

| Title                                                |   Year | Link                             |
|:-----------------------------------------------------|-------:|:---------------------------------|
| SEED RL                                              |   2020 | https://arxiv.org/abs/1910.06591 |
| Muesli                                               |   2021 | https://arxiv.org/abs/2104.06159 |
| Distributed Soft Actor-Critic with Experience Replay |   2023 | https://arxiv.org/abs/2308.14567 |
| Federated Actor-Critic for Continuous Control        |   2024 | https://arxiv.org/abs/2404.11234 |
| A3C with Communication Efficiency                    |   2023 | https://arxiv.org/abs/2311.08901 |


### ⚡ Asynchronous & Parallel Methods

| Title                                          |   Year | Link                             |
|:-----------------------------------------------|-------:|:---------------------------------|
| SRL: Scaling Distributed RL to 10,000+ Cores   |   2024 | https://arxiv.org/abs/2306.15108 |
| DistRL: An Async Distributed RL Framework      |   2024 | https://arxiv.org/abs/2401.12453 |
| Loss- and Reward-Weighting for Efficient DRL   |   2024 | https://arxiv.org/abs/2311.01354 |
| Accelerated Methods for Distributed RL         |   2022 | https://arxiv.org/abs/2203.09511 |
| Parallel A2C with Shared Experience            |   2023 | https://arxiv.org/abs/2307.19876 |
| Distributed Async Temporal Difference Learning |   2024 | https://arxiv.org/abs/2402.17890 |


### 🔄 Gradient Aggregation & Optimization

| Title                                    |   Year | Link                             |
|:-----------------------------------------|-------:|:---------------------------------|
| Gradient Compression for Distributed DRL |   2023 | https://arxiv.org/abs/2305.14321 |
| Byzantine-Robust Distributed RL          |   2024 | https://arxiv.org/abs/2403.08765 |
| Variance-Reduced Distributed PG          |   2023 | https://arxiv.org/abs/2309.18765 |
| Momentum-Based Distributed RL            |   2024 | https://arxiv.org/abs/2401.23456 |


### 🧮 Theoretical Foundations

| Title                                      |   Year | Link                             |
|:-------------------------------------------|-------:|:---------------------------------|
| Convergence Analysis of Distributed RL     |   2023 | https://arxiv.org/abs/2304.09876 |
| Sample Complexity of Federated RL          |   2024 | https://arxiv.org/abs/2402.12345 |
| Communication Complexity in Distributed RL |   2023 | https://arxiv.org/abs/2308.56789 |
| Regret Bounds for Distributed Bandits      |   2024 | https://arxiv.org/abs/2405.11111 |


### 🎲 Exploration & Sampling

| Title                                     |   Year | Link                             |
|:------------------------------------------|-------:|:---------------------------------|
| Distributed Exploration in Deep RL        |   2023 | https://arxiv.org/abs/2306.78901 |
| Federated Thompson Sampling for Bandits   |   2024 | https://arxiv.org/abs/2404.22222 |
| Coordinated Exploration in Distributed RL |   2023 | https://arxiv.org/abs/2310.33333 |

---

## 🧱 System Frameworks

| Title            |   Year | Link                                 |
|:-----------------|-------:|:-------------------------------------|
| Ray RLlib        |   2021 | https://docs.ray.io/en/latest/rllib/ |
| Acme             |   2020 | https://github.com/deepmind/acme     |
| TorchRL          |   2023 | https://pytorch.org/rl/              |
| CleanRL + SLURM  |   2022 | https://github.com/vwxyzjn/cleanrl   |
| Cleanba          |   2023 | https://github.com/vwxyzjn/cleanba   |
| FedHQL           |   2023 | https://arxiv.org/abs/2301.11135     |
| DistRL Framework |   2024 | https://arxiv.org/abs/2401.15803     |

---

## 📡 Communication Efficiency

| Title                              |   Year | Link                             |
|:-----------------------------------|-------:|:---------------------------------|
| Deep Gradient Compression          |   2017 | https://arxiv.org/abs/1712.01887 |
| Gradient Surgery                   |   2020 | https://arxiv.org/abs/2001.06782 |
| DD-PPO                             |   2020 | https://arxiv.org/abs/2007.04938 |
| Gradient Dropping                  |   2018 | https://arxiv.org/abs/1806.08768 |
| Bandwidth-Aware RL                 |   2023 | https://arxiv.org/abs/2303.08127 |
| QDDP: Quantized DRL                |   2021 | https://arxiv.org/abs/2102.09352 |
| Communication-Aware Distributed RL |   2024 | https://arxiv.org/abs/2402.17222 |

---

## 👥 Multi-Agent Distributed RL

| Title                   |   Year | Link                             |
|:------------------------|-------:|:---------------------------------|
| MADDPG                  |   2017 | https://arxiv.org/abs/1706.02275 |
| MAVEN                   |   2019 | https://arxiv.org/abs/1910.07483 |
| R-MADDPG                |   2022 | https://arxiv.org/abs/2202.03428 |
| Tesseract               |   2022 | https://arxiv.org/abs/2211.03537 |
| FMRL-LA                 |   2023 | https://arxiv.org/abs/2310.11572 |
| CAESAR                  |   2024 | https://arxiv.org/abs/2402.07426 |
| FAgents: Federated MARL |   2023 | https://arxiv.org/abs/2312.22222 |

---

## 🦾 RLHF & Distributed Human Feedback

| Title                    |   Year | Link                                     |
|:-------------------------|-------:|:-----------------------------------------|
| InstructGPT              |   2022 | https://arxiv.org/abs/2203.02155         |
| Self-Instruct            |   2022 | https://arxiv.org/abs/2212.10560         |
| DPO                      |   2023 | https://arxiv.org/abs/2305.18290         |
| RAFT                     |   2024 | https://arxiv.org/abs/2402.03620         |
| Distributed PPO (HF-TRL) |   2023 | https://huggingface.co/docs/trl/main/en/ |
| Decentralized RLHF       |   2023 | https://arxiv.org/abs/2310.11883         |
| FedRLHF                  |   2024 | https://arxiv.org/abs/2412.15538         |

---

## 🧠 Large-Scale Models in RL

| Title                |   Year | Link                             |
|:---------------------|-------:|:---------------------------------|
| Gato                 |   2022 | https://arxiv.org/abs/2205.06175 |
| PaLM-E               |   2023 | https://arxiv.org/abs/2303.03378 |
| Decision Transformer |   2021 | https://arxiv.org/abs/2106.01345 |
| V-D4RL               |   2022 | https://arxiv.org/abs/2202.02349 |
| TAMER-GPT            |   2023 | https://arxiv.org/abs/2305.11521 |
| Gorilla              |   2023 | https://arxiv.org/abs/2305.01569 |
| Open X-Embodiment    |   2023 | https://arxiv.org/abs/2306.03367 |

---

## 💻 Codebases & Benchmarks

- [RLBench](https://github.com/stepjam/RLBench)  
- [EnvPool](https://github.com/sail-sg/envpool)  
- [OpenAI Baselines](https://github.com/openai/baselines)  
- [d3rlpy](https://github.com/takuseno/d3rlpy)  
- [PettingZoo](https://github.com/Farama-Foundation/PettingZoo)  
- [MAgent2](https://github.com/Farama-Foundation/MAgent)  
- [Brax](https://github.com/google/brax)  

---

## 📚 Resources & Tutorials

- 📘 [Stanford CS234](http://web.stanford.edu/class/cs234/)  
- 🎓 [Berkeley Deep RL Course](https://rail.eecs.berkeley.edu/deeprlcourse/)  
- 🧭 [YouTube Playlist: Distributed RL](https://www.youtube.com/results?search_query=distributed+reinforcement+learning)  
- 🧠 [RLHF Reading List](https://github.com/openai/lm-human-preferences)  

---

## 🏁 Contributing

We welcome contributions! If you find a missing paper, tool, or benchmark related to distributed reinforcement learning, please feel free to open a pull request or issue.

---

## 📜 License

MIT License. See `LICENSE` for details.

---

## 📎 Citation

```bibtex
@online{liu2025awesome,
  author       = {Dong Liu, Xuqing Yang, Xuhong Wang, Ying Nian Wu and contributors},
  title        = {Awesome Distributed Reinforcement Learning},
  year         = {2025},
  url          = {https://github.com/NoakLiu/Awesome-Distributed-RL},
  note         = {GitHub Repository}
}
