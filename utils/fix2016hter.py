from pathlib import Path

data_folder= Path.cwd().parent / Path('data')
un_data_folder= Path.cwd().parent / Path('unused_data')
with open( un_data_folder / "wmt16" / "train.hter", 'r') as handle:
    lines = handle.readlines()
    lines = [str(float(i.strip()) /100) + '\n' for i in lines]
with open( data_folder / "wmt16" / "train.hter", 'w') as handle:
    handle.writelines(lines)

with open( un_data_folder / "wmt16" / "dev.hter", 'r') as handle:
    lines = handle.readlines()
    lines = [str(float(i.strip()) /100) + '\n' for i in lines]
with open( data_folder / "wmt16" / "dev.hter", 'w') as handle:
    handle.writelines(lines)
