from pathlib import Path
import random

data_folder= Path.cwd().parent / Path('data')


from collections import defaultdict
train_dict = defaultdict(list)
dev_dict = defaultdict(list)
test_dict = defaultdict(list)
for wmt in data_folder.iterdir():
    if wmt.name == "combined_data":
        continue
    for data in wmt.iterdir():
        file_name = data.name
        if file_name.startswith('train') and data.suffix in ['.src', '.mt', '.hter']:
            train_dict[wmt.name].append(data)
        elif file_name.startswith('dev') and data.suffix in ['.src', '.mt', '.hter']:
            dev_dict[wmt.name].append(data)
        elif file_name.startswith('test') and data.suffix in ['.src', '.mt', '.hter']:
            test_dict[wmt.name].append(data)
        else:
            pass

looper = zip(['train', 'dev', 'test'], [train_dict, dev_dict, test_dict])
seen = set()

# out_dict=defaultdict(defaultdict(list))
out_dict = { 'train' : defaultdict(list),
             'dev' : defaultdict(list),
             'test' : defaultdict(list)
             }
for category, paths in looper:
    for i in paths:
        src, hter, mt = sorted(paths[i], key=(lambda x: x.name[::-1]))
        with open(src) as src_file, open(hter) as hter_file, open(mt) as mt_file:

            index_list = [i for i in range(1000)]
            random.shuffle(index_list)
            test_indices = index_list[:203]
            train_indices = index_list[:746]
            dev_indices = index_list[746:]

            for index,  (src_line, hter_line, mt_line) in enumerate(zip(src_file, hter_file, mt_file)):
                if src_line in seen or len(src_line) < 4:
                    continue
                elif category == 'dev' and index in test_indices:
                    out_dict['test']['src'].append(src_line)
                    out_dict['test']['hter'].append(hter_line)
                    out_dict['test']['mt'].append(mt_line)
                elif category == 'dev' and index in train_indices:
                    out_dict['train']['src'].append(src_line)
                    out_dict['train']['hter'].append(hter_line)
                    out_dict['train']['mt'].append(mt_line)
                else:
                    seen.add(src_line)
                    out_dict[category]['src'].append(src_line)
                    out_dict[category]['hter'].append(hter_line)
                    out_dict[category]['mt'].append(mt_line)

print(len(out_dict['train']['src']))
print(len(out_dict['train']['mt']))
print(len(out_dict['train']['hter']))

print(len(out_dict['test']['src']))
print(len(out_dict['test']['mt']))
print(len(out_dict['test']['hter']))

print(len(out_dict['dev']['src']))
print(len(out_dict['dev']['mt']))
print(len(out_dict['dev']['hter']))

for category in out_dict.keys():
    for suffix in out_dict[category].keys():
        with open(data_folder / 'combined_data' / (category + '.' + suffix), 'w') as handle:
            handle.writelines(out_dict[category][suffix])
