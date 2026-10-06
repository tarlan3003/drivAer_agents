from tools.llm import call_llm
from config import MODEL_CRITIC
from orchestration.state import ResearchState
import json

CRITIC_SYSTEM = """You are the Critic Agent in an automotive aerodynamic design research system.
Your job is to rigorously review the findings produced by the other agents.

Check for:
1. Strength of evidence (is the conclusion well supported by the data?)
2. Important limitations (especially missing frontal area data)
3. Over-claiming or weak reasoning
4. Consistency between Data, ML and Physics findings
5. Whether the original user question was adequately answered

Be constructive but strict. Structure your feedback clearly:
- Strengths
- Weaknesses / Limitations
- Recommendations for the final report
"""

class CriticAgent:
    def run(self, state: ResearchState) -> ResearchState:
        user_prompt = f"""Original user query:
{state.user_query}

Dataset summary:
{json.dumps(state.data_summary, indent=2)}

Top designs (first 10):
{json.dumps(state.top_designs[:10], indent=2)}

ML Insights:
{state.ml_insights.get('llm_analysis', 'N/A') if state.ml_insights else 'N/A'}

Physics Explanation:
{state.physics_explanation or 'N/A'}

Please provide a critical review of the current findings.
"""

        feedback = call_llm(MODEL_CRITIC, CRITIC_SYSTEM, user_prompt, temperature=0.2, max_tokens=1200)

        state.critic_feedback = feedback
        state.add_message("Critic", "Completed critical review", {"feedback": feedback})
        return state