from pathlib import Path
import pandas as pd
from typing import Dict, Any

class DrivAerDataLoader:
    def __init__(self, csv_path: str = "data/DrivAerNetPlusPlus_Drag_8k.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self._load()

    def _load(self):
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.rename(columns={"Average Cd": "Cd"})

        # Parse Design ID
        parts = self.df["Design"].str.split("_", expand=True)
        self.df["body_type"] = parts[0]
        self.df["underbody"] = parts[1]
        self.df["wheel_config"] = parts[2]

        # Human labels
        body_map = {"F": "Fastback", "N": "Notchback", "E": "Estateback"}
        under_map = {"S": "Smooth", "D": "Detailed"}
        self.df["body_label"] = self.df["body_type"].map(body_map).fillna(self.df["body_type"])
        self.df["underbody_label"] = self.df["underbody"].map(under_map).fillna(self.df["underbody"])

        print(f"Loaded {len(self.df)} designs. Cd range: {self.df['Cd'].min():.4f} – {self.df['Cd'].max():.4f}")

    def get_summary_stats(self) -> Dict[str, Any]:
        return {
            "n_designs": int(len(self.df)),
            "cd_mean": float(self.df["Cd"].mean()),
            "cd_std": float(self.df["Cd"].std()),
            "cd_min": float(self.df["Cd"].min()),
            "cd_max": float(self.df["Cd"].max()),
            "cd_25": float(self.df["Cd"].quantile(0.25)),
            "cd_50": float(self.df["Cd"].quantile(0.50)),
            "cd_75": float(self.df["Cd"].quantile(0.75)),
        }

    def get_body_type_stats(self) -> Dict[str, Any]:
        stats = {}
        for label, group in self.df.groupby("body_label"):
            stats[label] = {
                "count": int(len(group)),
                "cd_mean": float(group["Cd"].mean()),
                "cd_min": float(group["Cd"].min()),
                "cd_25": float(group["Cd"].quantile(0.25)),
            }
        return stats

    def get_underbody_stats(self) -> Dict[str, Any]:
        stats = {}
        for label, group in self.df.groupby("underbody_label"):
            stats[label] = {
                "count": int(len(group)),
                "cd_mean": float(group["Cd"].mean()),
                "cd_min": float(group["Cd"].min()),
                "cd_25": float(group["Cd"].quantile(0.25)),
            }
        return stats

    def get_lowest_drag(self, top_k: int = 20) -> pd.DataFrame:
        cols = ["Design", "Cd", "body_label", "underbody_label", "wheel_config"]
        return self.df.nsmallest(top_k, "Cd")[cols].reset_index(drop=True)