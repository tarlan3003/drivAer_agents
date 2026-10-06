from tools.llm import call_llm
from config import MODEL_REPORT
from orchestration.state import ResearchState
import json
from datetime import datetime

REPORT_SYSTEM = """You are the Report Agent. Write a clear, accurate final report.

STRICT RULES:
- Only state what is directly supported by the data.
- Do NOT claim that frontal area was analysed.
- Do NOT say "strong correlation" or "critical" unless the numbers clearly justify it.
- Prefer careful language: "associated with", "predominant in the lowest-Cd designs", "appears beneficial".
- Always mention the missing frontal area as a limitation.
- Do not invent expansions for wheel configuration codes (WWC, WWS, WW). Just use the codes as they appear.

Required structure (use exactly these headings):

# Aerodynamic Design Analysis – DrivAerNet++

## 1. Executive Summary
## 2. Dataset Overview
## 3. Lowest-Drag Designs
## 4. Geometric Characteristics Associated with Low Drag
## 5. Support from the Available Data
## 6. Limitations
## 7. Recommendations

Use markdown. Be concise and precise.
"""

class ReportAgent:
    def run(self, state: ResearchState) -> ResearchState:
        user_prompt = f"""Original user query:
{state.user_query}

=== Data Summary ===
{json.dumps(state.data_summary, indent=2)}

=== Body Type Statistics ===
{json.dumps(state.data_summary.get("body_type_stats") if False else (state.messages[-1].data.get("body_type_stats") if state.messages else {}), indent=2) if False else "See data below"}

Full data package from Data Agent:
{json.dumps({
    "summary_stats": state.data_summary,
    "top_designs": state.top_designs[:12] if state.top_designs else [],
}, indent=2)}

=== ML Analysis ===
{state.ml_insights.get("llm_analysis", "") if state.ml_insights else ""}

=== Physics Explanation ===
{state.physics_explanation or ""}

=== Critic Feedback ===
{state.critic_feedback or ""}

Write the final report following the strict rules and required structure.
"""

        # Better: pass the rich data cleanly
        rich_data = {
            "summary_stats": state.data_summary,
            "top_designs": state.top_designs[:15] if state.top_designs else [],
        }
        # We also need body/underbody stats – they live in the last DataAgent message
        for msg in reversed(state.messages):
            if msg.agent == "DataAgent" and msg.data:
                rich_data["body_type_stats"] = msg.data.get("body_type_stats", {})
                rich_data["underbody_stats"] = msg.data.get("underbody_stats", {})
                break

        user_prompt = f"""Original user query:
{state.user_query}

Data package:
{json.dumps(rich_data, indent=2)}

ML Analysis:
{state.ml_insights.get("llm_analysis", "") if state.ml_insights else ""}

Physics Explanation:
{state.physics_explanation or ""}

Critic Feedback:
{state.critic_feedback or ""}

Write the final report following the STRICT RULES and the required structure.
"""

        report = call_llm(MODEL_REPORT, REPORT_SYSTEM, user_prompt, temperature=0.2, max_tokens=2200)

        state.final_report = report
        state.is_complete = True
        state.add_message("ReportAgent", "Final report generated")

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"final_report_{timestamp}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n✅ Report saved to {path}")

        return state