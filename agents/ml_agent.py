from tools.llm import call_llm
from config import MODEL_ML
from orchestration.state import ResearchState
from collections import Counter
import json

ML_SYSTEM = """You are the ML Agent for an automotive aerodynamic design system.
You receive a list of low-drag vehicle designs (with Design IDs and Cd values).
Your job is to analyse patterns in the Design IDs and produce clear, quantitative insights.

Focus on:
- Frequency of body types (F=Fastback, N=Notchback, E=Estate)
- Underbody type (S=Smooth, D=Detailed)
- Wheel / configuration codes (WWC, WWS, WW, etc.)
- Any repeated design numbers or clusters

Return a short structured analysis in plain text + a small JSON summary at the end.
Be factual and concise.
"""

class MLAgent:
    def run(self, state: ResearchState) -> ResearchState:
        if not state.top_designs:
            state.add_message("MLAgent", "No designs available to analyse")
            return state

        designs = state.top_designs
        design_ids = [d["Design"] for d in designs]
        cds = [d["Cd"] for d in designs]

        # Simple local statistics (no LLM needed)
        body_types = Counter(d.split("_")[0] for d in design_ids)
        underbodies = Counter()
        wheel_codes = Counter()

        for d in design_ids:
            parts = d.split("_")
            if len(parts) >= 2:
                underbodies[parts[1]] += 1
            if len(parts) >= 3:
                wheel_codes[parts[2]] += 1

        local_stats = {
            "body_type_counts": dict(body_types),
            "underbody_counts": dict(underbodies),
            "wheel_code_counts": dict(wheel_codes),
            "avg_cd_of_selection": sum(cds) / len(cds),
            "n_designs": len(designs)
        }

        # Ask the LLM for a readable interpretation
        user_prompt = f"""Here are the top low-drag designs:

{json.dumps(designs, indent=2)}

Local counts:
{json.dumps(local_stats, indent=2)}

Please analyse the patterns and summarise the dominant geometric characteristics that appear in the lowest-drag designs.
"""

        analysis = call_llm(MODEL_ML, ML_SYSTEM, user_prompt, temperature=0.2)

        state.ml_insights = {
            "local_stats": local_stats,
            "llm_analysis": analysis
        }
        state.add_message("MLAgent", "Completed design ID pattern analysis", state.ml_insights)
        return state