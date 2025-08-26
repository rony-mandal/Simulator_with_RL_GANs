# 🤖 Advanced AI-Driven Information Warfare Simulation

This project is an **internship research prototype** that simulates how information warfare narratives spread through social networks, and how **AI systems (Reinforcement Learning + GANs)** can adaptively generate and counter these narratives.  

It combines **agent-based modeling, reinforcement learning, and generative adversarial networks** to explore the dynamics of narrative competition, polarization, and counter-strategy design.  

---

## ✨ Features

- 🧠 **Reinforcement Learning Agents**: Adaptive agents learn optimal strategies using **Deep Q-Networks (DQN)**.  
- 🎭 **Narrative GAN**: Generates synthetic narratives and targeted counter-narratives.  
- 🕸️ **Agent-Based Simulation**: Realistic social network structure with influencers, skeptics, bots, and counter-agents.  
- ⚡ **Dynamic Events**: Crisis events (viral content, government statements, media reports) shift the narrative landscape.  
- 📊 **Interactive Dashboard**: Powered by **Streamlit + Plotly** for real-time simulation visualization.  
- 📈 **Advanced Analytics**: Tracks narrative spread, agent performance, network evolution, and polarization.  

---

## 📂 Project Structure

```bash
.
├── app.py                     # Streamlit dashboard to run & visualize simulations
├── processing/
│   └── narrative_processor.py # Preprocessing: embeddings + sentiment analysis
├── simulation/
│   ├── agents.py              # Basic agent behaviors (influencers, skeptics, regulars)
│   ├── rl_agent.py            # Reinforcement Learning agent with Deep Q-Network
│   ├── narrative_gan.py       # GAN + optional GPT-2 narrative generation system
│   └── model.py               # EnhancedNarrativeModel (simulation engine)
├── data/                      # CSV narrative datasets (war, health, economy, elections, etc.)
├── simulation_results/        # Logs and outputs from simulations
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

This will launch the dashboard in your browser.  

### Simulation Options

- **Data Input**: Choose *Manual Input* (type your own narratives) or *Preloaded Scenarios* (CSV datasets).  
- **AI Features**: Enable/disable RL, GAN-based narrative generation, and counter-narratives.  
- **Parameters**: Adjust number of agents, simulation steps, logging, and visualization frequency.  

---

## 📊 Example Simulation Flow

1. **Narratives Loaded**: The system ingests initial narratives (with embeddings + sentiment).  
2. **Agent Initialization**: Agents are created with roles: Influencer, Regular, Skeptic, Bot, Counter-Agent.  
3. **RL Dynamics**: Adaptive agents choose actions (spread, counter, form/break connections) via DQN.  
4. **GAN Dynamics**: GAN injects new narratives or generates counter-narratives when harmful narratives dominate.  
5. **Crisis Events**: Sudden shocks like viral posts or official statements alter dynamics.  
6. **Visualization**: Real-time dashboards show belief adoption, network structure, polarization, and agent learning curves.  

---

## 🔬 Modules Explained

- **`app.py`**  
  Streamlit-based UI for running simulations and visualizing results with Plotly charts.  

- **`processing/narrative_processor.py`**  
  Loads CSV scenarios or manual narratives, computes embeddings (`sentence-transformers`) and sentiment (`VADER`).  

- **`simulation/agents.py`**  
  Defines **NarrativeAgent**, a rule-based agent with type-specific influence and spread behavior.  

- **`simulation/rl_agent.py`**  
  Implements **AdaptiveNarrativeAgent** using **Deep Q-Networks** in PyTorch.  
  - Action space: spread, counter, form/break connections, ignore.  
  - Agents learn based on rewards tailored to their roles (influencer, skeptic, bot, counter-agent).  

- **`simulation/narrative_gan.py`**  
  GAN for narrative generation (Generator + Discriminator).  
  - Trains on scenario-specific narratives.  
  - Optional GPT-2 integration for advanced text generation.  
  - Can produce adaptive counter-narratives.  

- **`simulation/model.py`**  
  The **EnhancedNarrativeModel**, integrating all components:  
  - Creates agent population with realistic distribution.  
  - Builds scale-free social network (via NetworkX).  
  - Simulates narrative spread, crisis events, GAN injections.  
  - Tracks advanced metrics: entropy, polarization, phase transitions.  

---

## 📈 Results (Sample)

- Narrative spread curves (believers per narrative over time).  
- Heatmaps of narrative competition.  
- Agent learning curves (average reward by type).  
- Network structure metrics (density, clustering).  
- Crisis event logs and counter-narratives.  

*(You can insert screenshots of your Streamlit dashboard here for visual appeal.)*  

---

## 📚 Datasets

- `psyops_narratives.csv` (War/Conflict)  
- `economic_narratives.csv` (Economic crisis)  
- `health_narratives.csv` (Pandemics, outbreaks)  
- `election_narratives.csv` (Political campaigns)  
- `climate_narratives.csv` (Climate change)  
- `tech_narratives.csv` (AI & technology debates)  

Each dataset contains narrative texts (with optional sentiment).  

---

## 🔮 Future Work

- Integrate more sophisticated **transformer-based RL policies**.  
- Improve GAN training with larger datasets.  
- Add **visual network animations** of narrative diffusion.  
- Benchmark effectiveness of counter-narratives under different conditions.  

---
