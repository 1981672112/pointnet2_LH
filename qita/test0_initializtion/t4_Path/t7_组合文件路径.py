from pathlib import Path

path_str = Path(r"/usr/HinGwenWoong/")
path_str_join = path_str.joinpath("demo.py")
print(path_str_join)
