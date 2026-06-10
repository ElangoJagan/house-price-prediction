from src.utils import Utils

# Test reading config
config = Utils.read_yaml("config/config.yaml")
params = Utils.read_yaml("config/params.yaml")

print(config)
print("---")
print(params)