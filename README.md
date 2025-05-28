# 🧠🔥 Awesome Distributed Reinforcement Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of 🔥 **Distributed Reinforcement Learning** papers and systems. Covering scalable algorithms, system frameworks, multi-agent setups, large models, communication strategies, and RLHF — **130+ papers** and counting, maintained by [Dong Liu](https://github.com/NoakLiu).

---

## 🗂️ Table of Contents

- [📌 Key Surveys and Overviews](#📌-key-surveys-and-overviews)
- [🚀 Algorithms & Theoretical Advances](#🚀-algorithms--theoretical-advances)
  - [🔄 Actor-Critic & On-Policy](#🔄-actor-critic--on-policy)
  - [🧊 Off-Policy & Replay-Centric](#🧊-off-policy--replay-centric)
  - [🧠 Federated RL](#🧠-federated-rl)
  - [📈 Scalable Async Methods](#📈-scalable-async-methods)
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

### 🔄 Actor-Critic & On-Policy

- [IMPALA](https://arxiv.org/abs/1802.01561)  
- [SEED RL](https://arxiv.org/abs/1910.06591)  
- [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)  
- [Scalable On-Policy RL via Importance Weights](https://arxiv.org/abs/2109.08765)  

### 🧊 Off-Policy & Replay-Centric

- [Ape-X](https://arxiv.org/abs/1803.00933)  
- [R2D3](https://arxiv.org/abs/1910.01523)  
- [Reverb](https://arxiv.org/abs/1911.09844)  

### 🧠 Federated RL

- [FEDORA](https://arxiv.org/abs/2305.03097)  
- [FedRLHF](https://arxiv.org/abs/2412.15538)  
- [Asynchronous FedPG](https://arxiv.org/abs/2410.07965)  
- [Federated Natural Policy Gradient](https://arxiv.org/abs/2311.00201)  
- [Heterogeneous FedRL](https://arxiv.org/abs/2401.15273)  

### 📈 Scalable Async Methods

- [SRL](https://arxiv.org/abs/2306.02835)  
- [DistRL](https://arxiv.org/abs/2401.15803)  
- [Sample Factory 2.0](https://arxiv.org/abs/2109.12908)  
- [GA3C](https://arxiv.org/abs/1611.06256)  

---

## 🧱 System Frameworks

- [Ray RLlib](https://docs.ray.io/en/latest/rllib/)  
- [Acme](https://github.com/deepmind/acme)  
- [CleanRL + SLURM](https://github.com/vwxyzjn/cleanrl)  
- [TorchRL](https://pytorch.org/rl/)  
- [Cleanba](https://github.com/vwxyzjn/cleanba)  
- [FedHQL](https://arxiv.org/abs/2301.11135)  

---

## 📡 Communication Efficiency

- [Deep Gradient Compression](https://arxiv.org/abs/1712.01887)  
- [Gradient Surgery](https://arxiv.org/abs/2001.06782)  
- [DD-PPO](https://arxiv.org/abs/2007.04938)  
- [Gradient Dropping](https://arxiv.org/abs/1806.08768)  
- [Bandwidth-Aware RL](https://arxiv.org/abs/2303.08127)  
- [QDDP](https://arxiv.org/abs/2102.09352)  

---

## 👥 Multi-Agent Distributed RL

- [MADDPG](https://arxiv.org/abs/1706.02275)  
- [MAVEN](https://arxiv.org/abs/1910.07483)  
- [R-MADDPG](https://arxiv.org/abs/2202.03428)  
- [Tesseract](https://arxiv.org/abs/2211.03537)  
- [FMRL-LA](https://arxiv.org/abs/2310.11572)  
- [CAESAR](https://arxiv.org/abs/2402.07426)  

---

## 🦾 RLHF & Distributed Human Feedback

- [InstructGPT](https://arxiv.org/abs/2203.02155)  
- [Self-Instruct](https://arxiv.org/abs/2212.10560)  
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290)  
- [RAFT](https://arxiv.org/abs/2402.03620)  
- [Distributed PPO (HF-TRL)](https://huggingface.co/docs/trl/main/en/)  
- [Decentralized RLHF](https://arxiv.org/abs/2310.11883)  

---

## 🧠 Large-Scale Models in RL

- [Gato](https://arxiv.org/abs/2205.06175)  
- [PaLM-E](https://arxiv.org/abs/2303.03378)  
- [Decision Transformer](https://arxiv.org/abs/2106.01345)  
- [V-D4RL](https://arxiv.org/abs/2202.02349)  
- [TAMER-GPT](https://arxiv.org/abs/2305.11521)  
- [Gorilla](https://arxiv.org/abs/2305.01569)  

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
