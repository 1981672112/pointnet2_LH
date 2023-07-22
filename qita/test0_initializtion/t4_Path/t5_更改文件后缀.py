from pathlib import Path

path_str = Path(r"/usr/HinGwenWoong/demo.py")
path_suffix = path_str.with_suffix(".json")
print(path_suffix)
# /usr/HinGwenWoong/demo.json
