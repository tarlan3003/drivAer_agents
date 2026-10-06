# Agentic AI for Automotive Aerodynamic Design

A multi-agent system that investigates vehicle designs for **low aerodynamic drag** using the **DrivAerNet++** dataset.

Given a natural-language request such as:

> “Find vehicle designs that achieve low aerodynamic drag while maintaining a reasonable frontal area. Explain which geometric characteristics appear to contribute to the result and validate the recommendation against the available simulation data.”

the system autonomously plans, analyses, critiques, and reports its findings.

---

## Dataset

This project uses the **DrivAerNet++** dataset:

> Elrefaie, M., Morar, F., Dai, A., & Ahmed, F. (2024).  
> **DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks**.  
> *NeurIPS 2024 Datasets and Benchmarks Track*.  
> [arXiv:2406.09624](https://arxiv.org/abs/2406.09624)  
> GitHub: [https://github.com/Mohamedelrefaie/DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet)

We currently use the publicly available drag coefficient CSV (7,713 designs).  
Frontal area and the full set of 26 geometric parameters are not yet included, which the system explicitly acknowledges as a limitation.

**License note**: DrivAerNet++ is released under CC BY-NC 4.0 (non-commercial research/education only).

---

## Architecture

```
                    USER
                      │
                      ▼
               ┌──────────────┐
               │ Planner Agent│
               └──────┬───────┘
                      │
          ┌───────────┼────────────┐
          ▼           ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌─────────────┐
   │ Data Agent │ │ ML Agent │ │ Physics     │
   │            │ │          │ │ Analyst     │
   └─────┬──────┘ └────┬─────┘ └──────┬──────┘
         │             │              │
         └─────────────┼──────────────┘
                       ▼
                ┌─────────────┐
                │ Critic Agent│
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │Report Agent │
                └─────────────┘
```

### How the agents work together

| Agent              | Role                                                                 | Model (default)   |
|--------------------|----------------------------------------------------------------------|-------------------|
| **Planner**        | Breaks the user query into a concrete sequence of steps              | `gpt-4o`          |
| **Data Agent**     | Loads the DrivAerNet++ CSV, computes statistics, returns lowest-Cd designs and breakdowns by body type / underbody | deterministic + light LLM |
| **ML Agent**       | Analyses patterns in Design IDs (body type, underbody, wheel config) | `gpt-4o-mini`     |
| **Physics Analyst**| Explains the observed patterns using aerodynamic principles          | `gpt-4o`          |
| **Critic**         | Reviews the findings for evidence strength, consistency and limitations | `gpt-4o`       |
| **Report Agent**   | Writes a structured, cautious final report that answers the original question | `gpt-4o-mini` |

All agents share a common `ResearchState` object.  
Each agent reads from it, writes its results back, and the next agent builds on the previous work. The Critic forces the Report Agent to stay honest about missing data (especially frontal area).

---

## Project Structure

```
drivAer_agents/
├── agents/
│   ├── planner.py
│   ├── data_agent.py
│   ├── ml_agent.py
│   ├── physics_agent.py
│   ├── critic.py
│   └── report_agent.py
├── tools/
│   ├── data_loader.py
│   └── llm.py
├── orchestration/
│   ├── state.py
│   └── runner.py
├── data/
│   └── DrivAerNetPlusPlus_Drag_8k.csv   # (not committed)
├── prompts/                             # (optional)
├── config.py
├── main.py
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone <your-repo-url>
cd drivAer_agents

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install openai pandas pydantic python-dotenv

cp .env.example .env
# edit .env and put your OpenAI API key
```

Place the drag CSV in `data/DrivAerNetPlusPlus_Drag_8k.csv`.

---

## Usage

```bash
python main.py
```

The system will:
1. Create a plan
2. Query the dataset
3. Analyse design patterns
4. Provide aerodynamic explanations
5. Critique its own findings
6. Generate a final Markdown report (`final_report_YYYYMMDD_HHMMSS.md`)

---

## Current Limitations

- Frontal area data is not yet available → the system cannot enforce “reasonable frontal area”.
- Only scalar drag coefficients + Design ID categories are used.
- The full 26 geometric parameters and high-fidelity CFD fields are not loaded.

The agents are designed to surface these limitations clearly instead of hallucinating results.

---

## Citation

If you use this project, please cite both the original dataset and this work:

```bibtex
@inproceedings{elrefaie2024drivaernetplusplus,
  title={DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks},
  author={Elrefaie, Mohamed and Morar, Florin and Dai, Angela and Ahmed, Faez},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

---

## License
  
Dataset: CC BY-NC 4.0 (see DrivAerNet++ license)
