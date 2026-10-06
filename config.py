import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model choices
MODEL_PLANNER = "gpt-4o"
MODEL_DATA = "gpt-4o-mini"
MODEL_ML = "gpt-4o-mini"
MODEL_PHYSICS = "gpt-4o"
MODEL_CRITIC = "gpt-4o"
MODEL_REPORT = "gpt-4o-mini"

# Data
DATA_CSV = "data/DrivAerNetPlusPlus_Drag_8k.csv"