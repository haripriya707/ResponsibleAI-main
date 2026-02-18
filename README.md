# 🚗 Emotion-Augmented Reinforcement Learning for Safe Autonomous Driving

A Responsible AI framework that enhances Deep Reinforcement Learning (DRL) for autonomous driving by incorporating emotion-inspired safety modulation to improve risk-awareness and decision-making under uncertainty.

Built using the CARLA simulator, Stable-Baselines3, and hybrid visual–numerical state representations.

---

## 📌 Overview

This project implements DRL agents for autonomous driving in realistic urban environments using CARLA.

Unlike traditional RL agents that purely maximize reward, this system introduces:

**Emotion-Augmented Reinforcement Learning** — Inspired by human fear and risk perception to promote safer driving behavior.

The objective is to develop agents that are not only efficient, but responsible and safety-aware.

---

## 🧠 Key Contributions

- Implementation of PPO, DDPG, and SAC for continuous vehicle control
- Emotion-inspired risk modeling integrated into the reward function
- Safety-aware action modulation
- Visual representation learning using a Variational Autoencoder (VAE)
- Multi-input policy combining numerical and visual states
- Comprehensive evaluation framework with video recording and safety metrics

---

## 🏗 System Architecture

### 1. Simulation Environment

- Simulator: CARLA 0.9.13
- Urban driving scenarios
- Traffic light interaction
- Waypoint-based route planning
- Optional RGB camera observations

---

### 2. Reinforcement Learning Algorithms

Implemented using Stable-Baselines3 (PyTorch backend):

- PPO (Primary algorithm)
- DDPG
- SAC

Training configuration:
- 10M timesteps
- Model checkpointing every 1M steps
- TensorBoard logging
- CUDA support for GPU acceleration

---

### 3. Emotion-Augmented Safety Module (Core Innovation)

The emotion module models a dynamic risk signal based on driving context.

Fear intensity increases when:

- Speed exceeds safe threshold
- Distance to obstacles decreases
- Lane deviation increases
- Traffic rules are violated
- Environmental uncertainty increases

The emotion signal is integrated into the RL loop to:

- Modify reward penalties
- Scale aggressive actions
- Trigger early episode termination
- Penalize high-risk trajectories

This results in a risk-sensitive policy rather than pure reward maximization.

---

### 4. State Representation

Supports multiple state configurations:

Basic:
- Steering
- Throttle
- Speed
- Maneuver

Enhanced:
- Waypoints
- Distance to goal
- Traffic light state

Visual:
- RGB camera → VAE → latent embedding

---

### 5. Variational Autoencoder (VAE)

Used for:
- Compressing high-dimensional RGB observations
- Learning compact latent representations
- Improving RL sample efficiency

Implemented using TensorFlow 2.11.

---

## 📂 Project Structure

```
ResponsibleAI/
│
├── train.py
├── eval.py
├── config.py
├── utils.py
├── eval_plots.py
│
├── carla_env/
│   ├── carla_route_env.py
│   ├── rewards.py
│   ├── state_commons.py
│   ├── wrappers.py
│   └── navigation/
│
└── vae/
    ├── train_vae.py
    ├── models.py
    └── utils/
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/ResponsibleAI.git
cd ResponsibleAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Core Dependencies:

- CARLA 0.9.13
- Stable-Baselines3 1.6.2
- PyTorch 1.13.1
- TensorFlow 2.11.0
- Gym 0.21.0
- OpenCV 4.7.0

---

## 🚀 Training

Train an RL agent:

```bash
python train.py --algo ppo --reward emotion_augmented
```

Options:

```
--algo {ppo, ddpg, sac}
--state {basic, enhanced, visual}
--reward {baseline, safety, emotion_augmented}
```

---

## 📊 Evaluation

```bash
python eval.py --model path/to/model.zip
```

Outputs include:

- Episode reward
- Collision count
- Lane invasion rate
- Traffic light violations
- Goal completion rate
- Driving video recordings

---

## 📈 Metrics Evaluated

- Average episode reward
- Collision frequency
- Lane deviation
- Traffic compliance
- Success rate
- Speed stability

Emotion-augmented agents demonstrate improved safety performance compared to baseline PPO agents.

---

## 🔬 Future Work

- Bayesian uncertainty modeling
- Risk-sensitive PPO (CVaR optimization)
- Distributional RL
- Multi-agent traffic interaction
- Sim-to-real transfer

---

## 👩‍💻 Author

Haripriya N  
B.Tech Computer Science and Engineering  
Interests: Reinforcement Learning, Safety-Critical ML

---

## ⭐ Why This Matters

Autonomous vehicles must be:

- Intelligent
- Efficient
- Safe
- Responsible

Emotion-augmented RL bridges the gap between performance optimization and safety-aware intelligence.
