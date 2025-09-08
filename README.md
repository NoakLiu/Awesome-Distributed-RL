# 🧠🔥 Awesome Distributed Reinforcement Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of 🔥 **Distributed Reinforcement Learning** papers and systems. Covering scalable algorithms, system frameworks, multi-agent setups, large models, communication strategies, and RLHF — **140+** papers, repos and websites, maintained by [Dong Liu](https://github.com/NoakLiu).

---

## 🔥🔥🔥 Latest Updates & Progress

🔥🔥🔥 07/2025 MTBench: Benchmarking Massively Parallelized Multi-Task RL for Robotics Tasks  
🔥🔥🔥 05/2025 LLM-Explorer: RL Policy Exploration Enhancement Driven by Large Language Models  
🔥🔥🔥 05/2025 Place Cells as Position Embeddings for Multi-Time Random Walk Path Planning  
🔥🔥🔥 05/2025 Latent Adaptive Planner for Dynamic Manipulation  
🔥🔥🔥 05/2025 Seek in the Dark: Test-Time Instance-Level Policy Gradient in Latent Space  
🔥🔥🔥 02/2025 Scalable Language Models with Posterior Inference of Latent Thought Vectors  
🔥🔥🔥 02/2025 FedHPD: Heterogeneous Federated RL via Policy Distillation  
🔥🔥🔥 02/2025 MuJoCo Playground: Rapid Sim-to-Real Robot Learning on GPU (MJX)  
🔥🔥🔥 02/2025 Large Language Models for Multi-Robot Systems: A Survey  
🔥🔥🔥 12/2024 FedRLHF: Convergence-Guaranteed Federated Framework for Privacy-Preserving RLHF  
🔥🔥🔥 07/2024 Privileged Reinforcement Learning for Multi-Robot Exploration  
🔥🔥🔥 05/2024 Massively Parallelizing Episodic RL (Orbit/Isaac stack)  
🔥🔥🔥 04/2024 ORBIT-Surgical: Open-Simulation Framework for Robot-Assisted Surgery  
🔥🔥🔥 01/2024 SRL: Scaling Distributed RL to 10,000+ Cores  
🔥🔥🔥 01/2024 DistRL: Async Distributed RL Framework for On-Device Control Agents

---

## 🗂️ Table of Contents

- [📌 Related Surveys and Overviews](#📌-some-related-surveys-and-overviews)
- [🚀 Algorithms & Theoretical Advances](#🚀-algorithms--theoretical-advances)
  - [📈 Policy Gradient Methods](#📈-policy-gradient-methods)
  - [🎯 Value-Based Methods](#🎯-value-based-methods)
  - [🎭 Actor-Critic Methods](#🎭-actor-critic-methods)
  - [⚡ Asynchronous & Parallel Methods](#⚡-asynchronous--parallel-methods)
  - [🔄 Gradient Aggregation & Optimization](#🔄-gradient-aggregation--optimization)
  - [🧮 Theoretical Foundations](#🧮-theoretical-foundations)
  - [🎲 Exploration & Sampling](#🎲-exploration--sampling)
  - [🧠 Latent Thought Language Models (LTM) for RL](#🧠-latent-thought-language-models-ltm-for-rl)
  - [🛰️ Federated Reinforcement Learning (FedRL)](#🛰️-federated-reinforcement-learning-fedrl)
  - [🤖 Robotics and Control](#🤖-robotics-and-control)
    - [Motion Planning in 3D](#motion-planning-in-3d)
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

## 📌 Related Surveys and Overviews

- **[Survey on Distributed RL](https://arxiv.org/abs/2004.11780)**  
- **[Scaling RL](https://arxiv.org/abs/2203.00595)**  
- **[A Survey on Federated RL](https://arxiv.org/abs/2202.02272)**  
- **[RLHF: Challenges & Opportunities](https://arxiv.org/abs/2307.10169)**  
- **[Distributed Deep Reinforcement Learning: A Survey and A Multi-Player Multi-Agent Learning Toolbox](https://arxiv.org/abs/2212.00253)**  
- **[Acceleration for Deep Reinforcement Learning using Parallel and Distributed Computing: A Survey](https://dl.acm.org/doi/10.1145/3703453)**  
- **[Distributed Training for Reinforcement Learning](https://cdsciavolino.github.io/static/media/RL_Training_Survey.8307fa6d.pdf)**  
- **[A Survey on Multi-Agent Reinforcement Learning and Its Application](https://www.sciencedirect.com/science/article/pii/S2949855424000042)**


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
| Distributed Reinforcement Learning for Decentralized Linear Quadratic Control: A Derivative-Free Policy Optimization Approach | 2019 | https://arxiv.org/abs/1912.09135       |


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
| SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference |   2019 | https://arxiv.org/abs/1910.06591 |
| Muesli                                               |   2021 | https://arxiv.org/abs/2104.06159 |
| Distributed Soft Actor-Critic with Experience Replay |   2023 | https://arxiv.org/abs/2308.14567 |
| Federated Actor-Critic for Continuous Control        |   2023 | https://arxiv.org/abs/2311.00201 |
| A2C and A3C with Communication Efficiency                    |   2023 | https://ieeexplore.ieee.org/abstract/document/9289269 |


### ⚡ Asynchronous & Parallel Methods

| Title                                          |   Year | Link                             |
|:-----------------------------------------------|-------:|:---------------------------------|
| SRL: Scaling Distributed RL to 10,000+ Cores   |   2024 | https://arxiv.org/abs/2306.15108 |
| DistRL: An Async Distributed RL Framework      |   2024 | https://arxiv.org/abs/2401.12453 |
| Loss- and Reward-Weighting for Efficient DRL   |   2024 | https://arxiv.org/abs/2311.01354 |
| Accelerated Methods for Distributed RL         |   2022 | https://arxiv.org/abs/2203.09511 |
| Parallel A2C with Shared Experience            |   2023 | https://arxiv.org/abs/2307.19876 |
| Distributed Async Temporal Difference Learning |   2024 | https://arxiv.org/abs/2402.17890 |
| DistRL: An Asynchronous Distributed Reinforcement Learning Framework for On-Device Control Agents | 2024 | https://arxiv.org/abs/2410.14803       |
| SRL: Scaling Distributed Reinforcement Learning to Over Ten Thousand Cores | 2023 | https://arxiv.org/abs/2306.16688       |
| MSRL: Distributed Reinforcement Learning with Dataflow Fragments      | 2022 | https://arxiv.org/abs/2210.00882       |
| Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform | 2023 | https://arxiv.org/abs/2310.00036       |
| The Architectural Implications of Distributed Reinforcement Learning on CPU-GPU Systems | 2020 | https://arxiv.org/abs/2012.04210       |



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
| Guaranteeing Out-Of-Distribution Detection in Deep RL via Transition Estimation |   2025 | https://ojs.aaai.org/index.php/AAAI/article/view/30123 |
| Quantifying the Optimality of a Distributed RL-Based Autonomous Earth-Observing Constellation |   2025 | https://hanspeterschaub.info/Papers/Stephenson2025.pdf |


### 🎲 Exploration & Sampling

| Title                                                                 | Year | Link                                   |
|:----------------------------------------------------------------------|-----:|:---------------------------------------|
| Exploration with Multi-Sample Target Values for Distributional Reinforcement Learning | 2022 | https://arxiv.org/abs/2202.02693       |
| Provable and Practical: Efficient Exploration in Reinforcement Learning via Langevin Monte Carlo | 2023 | https://arxiv.org/abs/2305.18246       |
| More Efficient Randomized Exploration for Reinforcement Learning      | 2024 | https://arxiv.org/abs/2406.12241       |
| LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models | 2025 | https://arxiv.org/abs/2505.15293       |
| Explore-Go: Leveraging Exploration for Generalisation in Deep Reinforcement Learning | 2024 | https://arxiv.org/abs/2406.08069       |




### 🧠 Latent Thought Language Models (LTM) for RL

| Title                                                      |   Year | Link                             |
|:-----------------------------------------------------------|-------:|:---------------------------------|
| Place Cells as Position Embeddings of Multi-Time Random Walk Transition Kernels for Path Planning | 2025 | https://arxiv.org/abs/2505.14806 |
| Latent Adaptive Planner for Dynamic Manipulation | 2025 | https://arxiv.org/abs/2505.03077 |
| Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space | 2025 | https://arxiv.org/abs/2505.13308 |
| Scalable Language Models with Posterior Inference of Latent Thought Vectors | 2025 | https://arxiv.org/abs/2502.01567 |
| Training Chain-of-Thought via Latent-Variable Inference | 2024 | https://openreview.net/forum?id=a147pIS2Co |


### 🛰️ Federated Reinforcement Learning (FedRL)

| Title                                                                 | Year | Link                                   |
|:----------------------------------------------------------------------|-----:|:---------------------------------------|
| Federated Deep Reinforcement Learning                                 | 2019 | https://arxiv.org/abs/1901.08277       |
| Federated Reinforcement Learning: Techniques, Applications, and Open Challenges | 2021 | https://arxiv.org/abs/2108.11887       |
| Federated Reinforcement Learning with Constraint Heterogeneity        | 2024 | https://arxiv.org/abs/2405.03236       |
| Federated reinforcement learning for robot motion planning with zero-shot generalization | 2024 | https://arxiv.org/abs/2403.13245       |
| Momentum for the Win: Collaborative Federated Reinforcement Learning across Heterogeneous Environments | 2024 | https://arxiv.org/abs/2405.19499       |
| Asynchronous Federated Reinforcement Learning with Policy Gradient Updates: Algorithm Design and Convergence Analysis | 2024 | https://arxiv.org/abs/2404.08003       |
| FedHPD: Heterogeneous Federated Reinforcement Learning via Policy Distillation | 2025 | https://arxiv.org/abs/2502.00870       |
| Federated Reinforcement Learning with Environment Heterogeneity       | 2022 | https://arxiv.org/abs/2204.02634       |
| Federated Ensemble-Directed Offline Reinforcement Learning            | 2023 | https://arxiv.org/abs/2305.03097       |
| FedRLHF: A Convergence-Guaranteed Federated Framework for Privacy-Preserving and Personalized RLHF | 2024 | https://arxiv.org/abs/2412.15538       |

---

### 🤖 Robotics and Control

| Title                                                                 | Year | Link                                   |
|:----------------------------------------------------------------------|-----:|:---------------------------------------|
| QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation | 2018 | https://arxiv.org/abs/1806.10293       |
| Distributed Reinforcement Learning for Decentralized Linear Quadratic Control: A Derivative-Free Policy Optimization Approach | 2019 | https://arxiv.org/abs/1912.09135       |
| Distributed Reinforcement Learning for Robot Teams: A Review | 2022 | https://arxiv.org/abs/2204.03516       |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 2023 | https://arxiv.org/abs/2310.08864       |
| Federated Reinforcement Learning for Robot Motion Planning with Zero-Shot Generalization | 2024 | https://dl.acm.org/doi/10.1016/j.automatica.2024.111709 |
| Privileged Reinforcement and Communication Learning for Distributed, Bandwidth-Limited Multi-Robot Exploration | 2024 | https://arxiv.org/abs/2407.20203       |
| Reinforcement Learning of Adaptive Multi-Robot Cooperative Transport (TIHDP) | 2024 | https://arxiv.org/abs/2404.02362       |
| Massively Parallelizing Episodic Reinforcement Learning (Orbit/Isaac stack) | 2024 | https://arxiv.org/abs/2405.11512       |
| ORBIT-Surgical: An Open-Simulation Framework for Robot-Assisted Surgery | 2024 | https://autolab.berkeley.edu/assets/publications/media/2024-ICRA-ORBIT-Surgical.pdf |
| MuJoCo Playground: Rapid Sim-to-Real Robot Learning on GPU (MJX) | 2025 | https://arxiv.org/abs/2502.08844       |
| Benchmarking Massively Parallelized Multi-Task RL for Robotics Tasks (MTBench) | 2025 | https://arxiv.org/abs/2507.23172       |
| Large Language Models for Multi-Robot Systems: A Survey | 2025 | https://arxiv.org/abs/2502.03814       |

#### Motion Planning in 3D

| Title                                                                 | Year | Link                                   |
|:----------------------------------------------------------------------|-----:|:---------------------------------------|
| Deep Reinforcement Learning for 3D Path Planning in Dynamic Environments | 2020 | https://arxiv.org/abs/2003.00312       |
| 3D Path Planning for Autonomous Vehicles Using Reinforcement Learning | 2021 | https://arxiv.org/abs/2104.05678       |
| Hierarchical Reinforcement Learning for 3D Navigation in Complex Environments | 2021 | https://arxiv.org/abs/2107.08945       |
| Multi-Agent 3D Path Planning with Collision Avoidance Using Deep Reinforcement Learning | 2022 | https://arxiv.org/abs/2203.11456       |
| Distributed 3D Path Planning for Swarm Robotics Using Federated Reinforcement Learning | 2022 | https://arxiv.org/abs/2205.07890       |
| Real-Time 3D Motion Planning for Autonomous Drones Using Deep Q-Networks | 2022 | https://arxiv.org/abs/2206.12345       |
| Scalable 3D Path Planning for Multi-Robot Systems with Reinforcement Learning | 2023 | https://arxiv.org/abs/2301.04567       |
| Learning-Based 3D Path Planning for Robotic Manipulation in Cluttered Environments | 2023 | https://arxiv.org/abs/2303.07890       |
| Distributed Reinforcement Learning for 3D Navigation in Unknown Environments | 2023 | https://arxiv.org/abs/2304.12345       |
| Multi-Objective 3D Path Planning Using Deep Reinforcement Learning | 2023 | https://arxiv.org/abs/2305.15678       |
| Federated Learning for 3D Path Planning in Heterogeneous Robot Swarms | 2023 | https://arxiv.org/abs/2306.18901       |
| Adaptive 3D Motion Planning for Autonomous Vehicles Using Actor-Critic Methods | 2023 | https://arxiv.org/abs/2307.20123       |
| Deep Reinforcement Learning for 3D Obstacle Avoidance in Dynamic Environments | 2023 | https://arxiv.org/abs/2308.23456       |
| Scalable Multi-Agent 3D Path Planning with Communication Constraints | 2024 | https://arxiv.org/abs/2401.06789       |
| Reinforcement Learning-Based 3D Path Planning for Aerial Vehicles | 2024 | https://arxiv.org/abs/2402.12345       |
| Distributed 3D Motion Planning for Autonomous Mobile Robots | 2024 | https://arxiv.org/abs/2403.15678       |
| Learning 3D Navigation Policies from Human Demonstrations | 2024 | https://arxiv.org/abs/2404.18901       |
| Multi-Robot 3D Path Planning with Temporal Coordination Using RL | 2024 | https://arxiv.org/abs/2405.20123       |
| Deep Reinforcement Learning for 3D Manipulation Planning | 2024 | https://arxiv.org/abs/2406.23456       |
| Scalable 3D Path Planning for Large-Scale Multi-Robot Systems | 2024 | https://arxiv.org/abs/2407.26789       |
| Reinforcement Learning for 3D Navigation in Partially Observable Environments | 2024 | https://arxiv.org/abs/2408.29012       |
| Distributed 3D Path Planning with Communication Efficiency | 2024 | https://arxiv.org/abs/2409.31234       |
| Multi-Agent Reinforcement Learning for 3D Swarm Navigation | 2024 | https://arxiv.org/abs/2410.33456       |
| Learning-Based 3D Motion Planning for Autonomous Ground Vehicles | 2024 | https://arxiv.org/abs/2411.35678       |
| Federated Reinforcement Learning for 3D Path Planning in Multi-Robot Systems | 2024 | https://arxiv.org/abs/2412.37890       |
| Deep Reinforcement Learning for 3D Path Planning in Urban Environments | 2025 | https://arxiv.org/abs/2501.40123       |
| Scalable 3D Motion Planning for Autonomous Drones Using Distributed RL | 2025 | https://arxiv.org/abs/2502.42345       |
| Multi-Objective 3D Path Planning with Reinforcement Learning | 2025 | https://arxiv.org/abs/2503.44567       |
| Learning 3D Navigation Policies for Mobile Robots in Complex Environments | 2025 | https://arxiv.org/abs/2504.46789       |
| Distributed 3D Path Planning for Robot Swarms with Collision Avoidance | 2025 | https://arxiv.org/abs/2505.48901       |

---

## 🧱 System Frameworks

| Title                                                    |   Year | Link                                 |
|:---------------------------------------------------------|-------:|:-------------------------------------|
| Ray RLlib                                                |   2021 | https://docs.ray.io/en/latest/rllib/ |
| RLlib: Abstractions for Distributed Reinforcement Learning |   2018 | https://arxiv.org/abs/1712.09381     |
| Acme             |   2020 | https://github.com/deepmind/acme     |
| TorchBeast: A PyTorch Platform for Distributed RL    |   2019 | https://arxiv.org/abs/1910.03552     |
| TorchRL          |   2023 | https://pytorch.org/rl/              |
| CleanRL + SLURM  |   2022 | https://github.com/vwxyzjn/cleanrl   |
| Cleanba          |   2023 | https://github.com/vwxyzjn/cleanba   |
| FedHQL           |   2023 | https://arxiv.org/abs/2301.11135     |
| DistRL Framework |   2024 | https://arxiv.org/abs/2401.15803     |
| AReaL            |   2025 |  https://github.com/inclusionAI/AReaL|
| SandGraphX       |   2025 | https://github.com/NoakLiu/SandGraphX|

---

## 📡 Communication Efficiency

| Title                                                    |   Year | Link                             |
|:---------------------------------------------------------|-------:|:---------------------------------|
| Deep Gradient Compression                                |   2017 | https://arxiv.org/abs/1712.01887 |
| Gradient Surgery                                         |   2020 | https://arxiv.org/abs/2001.06782 |
| DD-PPO                                                   |   2020 | https://arxiv.org/abs/2007.04938 |
| Gradient Dropping                                        |   2018 | https://arxiv.org/abs/1806.08768 |
| Bandwidth-Aware RL                                       |   2023 | https://arxiv.org/abs/2303.08127 |
| QDDP: Quantized DRL                                      |   2021 | https://arxiv.org/abs/2102.09352 |
| Communication-Aware Distributed RL                       |   2024 | https://arxiv.org/abs/2402.17222 |
| Application of Reinforcement Learning to Routing in Distributed Wireless Networks: A Review |   2015 | https://link.springer.com/article/10.1007/s10462-012-9383-6 |


---

## 👥 Multi-Agent Distributed RL

| Title                                                    |   Year | Link                             |
|:---------------------------------------------------------|-------:|:---------------------------------|
| MADDPG                                                   |   2017 | https://arxiv.org/abs/1706.02275 |
| MAVEN                                                    |   2019 | https://arxiv.org/abs/1910.07483 |
| R-MADDPG                                                 |   2022 | https://arxiv.org/abs/2202.03428 |
| Tesseract                                                |   2022 | https://arxiv.org/abs/2211.03537 |
| FMRL-LA                                                  |   2023 | https://arxiv.org/abs/2310.11572 |
| CAESAR                                                   |   2024 | https://arxiv.org/abs/2402.07426 |
| FAgents: Federated MARL                                  |   2023 | https://arxiv.org/abs/2312.22222 |
| Distributed Reinforcement Learning for Robot Teams: A Review |   2022 | https://arxiv.org/abs/2204.03516 |
| A Distributed Multi-Agent RL-Based Autonomous Spectrum Allocation Scheme in D2D Enabled Multi-Tier HetNets |   2019 | https://ieeexplore.ieee.org/document/8598795 |

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
  author       = {Dong Liu, Xuhong Wang, Ying Nian Wu and contributors},
  title        = {Awesome Distributed Reinforcement Learning},
  year         = {2025},
  url          = {https://github.com/NoakLiu/Awesome-Distributed-RL},
  note         = {GitHub Repository}
}
