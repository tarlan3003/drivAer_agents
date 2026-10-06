from tools.llm import call_llm
from config import MODEL_PHYSICS
from orchestration.state import ResearchState
import json

PHYSICS_SYSTEM = """You are the Physics Analyst Agent specialising in automotive aerodynamics.
You receive:
- Summary statistics of the full DrivAerNet++ dataset
- A list of the lowest-drag designs
- Pattern analysis from the ML Agent

Your task is to explain, using sound aerodynamic principles, which geometric characteristics most likely contribute to the low drag values.

Key concepts you can use:
- Fastback vs Notchback vs Estate rear-end shapes and their effect on wake size / separation
- Smooth underbody (typical of EVs) vs detailed underbody (ICE) and ground-effect / underbody flow
- Wheel design and wheel-house aerodynamics
- Overall length, roof height, greenhouse angle, diffuser angle, rear window inclination (even if exact numbers are not present, reason about the categories)

Be precise, cite the observed patterns, and clearly separate strong evidence from plausible hypotheses.
Keep the answer structured and technical but readable.
"""

class PhysicsAgent:
    def run(self, state: ResearchState) -> ResearchState:
        if not state.top_designs or not state.ml_insights:
            state.add_message("PhysicsAgent", "Missing data or ML insights")
            return state

        user_prompt = f"""User original request:
{state.user_query}

Dataset summary:
{json.dumps(state.data_summary, indent=2)}

Top low-drag designs:
{json.dumps(state.top_designs[:12], indent=2)}

ML pattern analysis:
{state.ml_insights.get('llm_analysis', '')}

Local counts:
{json.dumps(state.ml_insights.get('local_stats', {}), indent=2)}

Please provide a clear aerodynamic explanation of the geometric characteristics that appear to drive the low drag results.
Also note the current limitation that frontal area data is not yet available.
"""

        explanation = call_llm(MODEL_PHYSICS, PHYSICS_SYSTEM, user_prompt, temperature=0.3, max_tokens=1800)

        state.physics_explanation = explanation
        state.add_message("PhysicsAgent", "Generated aerodynamic explanation", {"explanation": explanation})
        return state