from tools.data_loader import DrivAerDataLoader
from typing import Dict, Any

class DataAgent:
    def __init__(self, csv_path: str = "data/DrivAerNetPlusPlus_Drag_8k.csv"):
        self.loader = DrivAerDataLoader(csv_path)

    def run(self, top_k: int = 20) -> Dict[str, Any]:
        stats = self.loader.get_summary_stats()
        body_stats = self.loader.get_body_type_stats()
        under_stats = self.loader.get_underbody_stats()
        lowest = self.loader.get_lowest_drag(top_k=top_k)

        return {
            "summary_stats": stats,
            "body_type_stats": body_stats,
            "underbody_stats": under_stats,
            "top_low_drag_designs": lowest.to_dict(orient="records"),
            "note": "Frontal area data is not present in the current CSV. All rankings are by Cd only.",
            "agent": "DataAgent"
        }