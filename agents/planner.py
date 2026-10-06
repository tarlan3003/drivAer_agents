from tools.llm import call_llm
from config import MODEL_PLANNER
from orchestration.state import ResearchState
import json

PLANNER_SYSTEM = """You are the Planner Agent for an automotive aerodynamic design research system.
Your job is to break the user's request into a clear sequence of steps that the specialized agents can execute.

Available agents:
- DataAgent: loads the DrivAerNet++ dataset, returns summary stats and lowest-drag designs
- MLAgent: performs simple analysis / clustering / feature patterns on design IDs
- PhysicsAnalyst: explains geometric reasons for low drag using domain knowledge
- Critic: reviews findings for consistency and evidence quality
- ReportAgent: writes the final structured answer

Return ONLY a JSON list of steps, for example:
["DataAgent: get top 20 lowest Cd designs and summary stats",
 "MLAgent: analyze design ID patterns in the top designs",
 "PhysicsAnalyst: explain why these designs have low drag",
 "Critic: check the evidence",
 "ReportAgent: produce final answer"]
"""

class PlannerAgent:
    def run(self, state: ResearchState) -> ResearchState:
        user_prompt = f"""User query:
{state.user_query}

Create a short, practical plan (max 5-6 steps)."""

        raw = call_llm(MODEL_PLANNER, PLANNER_SYSTEM, user_prompt, temperature=0.1)
        
        # Simple extraction of the list
        try:
            # Find the JSON list in the response
            start = raw.find("[")
            end = raw.rfind("]") + 1
            plan = json.loads(raw[start:end])
        except Exception:
            plan = [
                "DataAgent: get top 15 lowest Cd designs and summary stats",
                "MLAgent: analyze design ID patterns",
                "PhysicsAnalyst: explain geometric contributors to low drag",
                "Critic: validate the findings",
                "ReportAgent: write final report"
            ]

        state.plan = plan
        state.add_message("Planner", f"Plan created with {len(plan)} steps", {"plan": plan})
        return state