from orchestration.state import ResearchState
from agents.planner import PlannerAgent
from agents.data_agent import DataAgent
from agents.ml_agent import MLAgent
from agents.physics_agent import PhysicsAgent
from agents.critic import CriticAgent
from agents.report_agent import ReportAgent
from datetime import datetime


class Runner:
    def __init__(self):
        self.planner = PlannerAgent()
        self.data_agent = DataAgent()
        self.ml_agent = MLAgent()
        self.physics_agent = PhysicsAgent()
        self.critic = CriticAgent()
        self.report_agent = ReportAgent()

    def run(self, user_query: str) -> ResearchState:
        state = ResearchState(user_query=user_query)

        # 1. Planner
        print("\n=== PLANNER ===")
        state = self.planner.run(state)
        for i, step in enumerate(state.plan, 1):
            print(f"{i}. {step}")

        # 2. Data Agent
        print("\n=== DATA AGENT ===")
        data_result = self.data_agent.run(top_k=20)
        # inside run(), after data_result = ...
        state.data_summary = data_result["summary_stats"]
        state.top_designs = data_result["top_low_drag_designs"]
        state.add_message("DataAgent", "Retrieved data", data_result)   # keep the full dict
        print(f"Cd mean: {state.data_summary['cd_mean']:.4f} | Top design: {state.top_designs[0]['Design']} (Cd={state.top_designs[0]['Cd']:.5f})")

        # 3. ML Agent
        print("\n=== ML AGENT ===")
        state = self.ml_agent.run(state)
        print(state.ml_insights["llm_analysis"])

        # 4. Physics Analyst
        print("\n=== PHYSICS ANALYST ===")
        state = self.physics_agent.run(state)
        print(state.physics_explanation)

        # 5. Critic
        print("\n=== CRITIC ===")
        state = self.critic.run(state)
        print(state.critic_feedback)

        # 6. Report Agent
        print("\n=== REPORT AGENT ===")
        state = self.report_agent.run(state)
        print(state.final_report)

        # Save final report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"final_report_{timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(state.final_report)
        print(f"\n✅ Full report saved to: {report_path}")

        return state