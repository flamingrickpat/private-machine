import datetime
import os
import yaml

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy import create_engine, text

# Define module-level variables with default empty values
companion_name = ""
user_name = ""
database_uri = ""
embedding_model = ""

cluster_split = 0.33

timestamp_format = ""

sysprompt = ""
char_card_3rd_person_neutral = ""
char_card_3rd_person_emotional = ""
sysprompt_addendum = ""

# Get absolute path to the config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    raise Exception("config.yaml doesn't exist!")

# Load YAML file
with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# Update global variables dynamically
global_map = {
    "companion_name": config_data.get("companion_name", ""),
    "user_name": config_data.get("user_name", ""),
    "database_uri": config_data.get("database_uri", ""),
    "embedding_model": config_data.get("embedding_model", ""),
    "cluster_split": config_data.get("cluster_split", 0.33),
    "timestamp_format": config_data.get("timestamp_format", ""),
    "sysprompt": config_data.get("sysprompt", ""),
    "char_card_3rd_person_neutral": config_data.get("char_card_3rd_person_neutral", ""),
    "char_card_3rd_person_emotional": config_data.get("char_card_3rd_person_emotional", ""),
    "sysprompt_addendum": config_data.get("sysprompt_addendum", ""),
}
globals().update(global_map)

# Extract model configurations
models = config_data.get("models", {})
model_mapping = config_data.get("model_mapping", {})

# Generate model_map dictionary
model_map = {}
for model_class, model_key in model_mapping.items():
    if model_key in models:
        model_map[model_class] = {
            "path": models[model_key]["path"],
            "layers": models[model_key]["layers"],
            "context": models[model_key]["context"],
            "last_n_tokens_size": models[model_key]["last_n_tokens_size"],
        }
    else:
        raise KeyError(f"Model key '{model_key}' in model_mapping not found in models section.")

# create db and activate vector extension
engine = create_engine(database_uri)
if not database_exists(engine.url):
    create_database(engine.url)

    with engine.connect() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        con.commit()


