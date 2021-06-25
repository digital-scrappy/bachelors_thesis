import pygmo
import optuna
import pickle
from collections import defaultdict
import sklearn
import statistics
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

def get_objectives(trials, MOO = True ):
    '''takes a list of trials and returns their objective values note that the Corr gets inverted'''
    objective_list = []
    for trial in trials:
        if MOO:
            trial_objectives = [ 1 - trial.user_attrs['test_pearson'], trial.values[1]]
        else:
            trial_objectives = [ 1 - trial.user_attrs['test_pearson'], trial.user_attrs['Flops']]

        objective_list.append(trial_objectives)

    return objective_list

class my_study:

    def __init__(self, seed, sampler_type, trials, objectives, study):
        self.seed = seed
        self.sampler_type = sampler_type
        self.trials = trials
        self.objective_list = objectives
        self.name = self.seed + '_' + self.sampler_type
        self.optuna_study = study
        self.ranked_fronts = []





        self.fronts = None
        self.pareto_optimal_front = None
        self.hv = None
        self.hv_history = []

    def __repr__(self):
        return self.name

    def non_dominated_sorting(self):
        # print(self.objective_list)

        self.fronts = pygmo.fast_non_dominated_sorting(self.objective_list)[0]
        for index, front in enumerate(self.fronts):
            self.ranked_fronts += [ (index, self.objective_list[i]) for i in front]





        self.pareto_optimal_front = [self.objective_list[i] for i in self.fronts[0]]
        # print(self.pareto_optimal_front)

        # self.dev_fronts = pygmo.fast_non_dominated_sorting(self.dev_objective_list)[0]

        # self.dev_pareto_optimal_front = [self.dev_objective_list[i] for i in self.dev_fronts[0]]
    def calculate_hv_history(self , ref_point):

        for i in range(1, len(self.objective_list)):
            if i == 1:
                pareto_front =[self.objective_list[0]]
            else:
                trial_subset = self.objective_list[:i]
                fronts = pygmo.fast_non_dominated_sorting(trial_subset)[0]
                pareto_front = [trial_subset[i] for i in fronts[0]]
            hv_i = pygmo.hypervolume(pareto_front)
            self.hv_history.append(hv_i.compute(ref_point))




    def calculate_hv(self, ref_point):
        hv = pygmo.hypervolume(self.pareto_optimal_front)
        self.hv = hv.compute(ref_point)


        # self.dev_hv = pygmo.hypervolume(self.dev_pareto_optimal_front)
        # print(f"dev:{self.dev_hv.compute(dev_ref_point)}")

base_path = Path("/home/jakob/src/siamese_attention_thesis/studies/")
seeds = [str(i) for i in range(3)]
sampler_names = ["RandomSampler", "MOTPESampler", "NSGAIISampler", "TPESampler"]
multi = ["RandomSampler", "MOTPESampler", "NSGAIISampler"]
trial_number = "100"




all_test_objectives = []
all_random_objectives = []
studies = []
for seed in seeds:
    for sampler in sampler_names:
            study_name = seed + "_" + sampler + "_" + trial_number + ".pkl"
            study_path = base_path / study_name

            with open(study_path, 'rb') as handle:
                study = pickle.load(handle)
                trials = study.trials
                moo = True if sampler in multi else False
                objectives = get_objectives(trials, moo)
                if sampler == 'RandomSampler':
                    all_random_objectives += objectives


                all_test_objectives += objectives
                studies.append(my_study(seed, sampler, trials, objectives, study))

all_hv = pygmo.hypervolume(all_test_objectives)
reference_point = all_hv.refpoint(1)
print(reference_point)

import  statistics
sum_hv = defaultdict(list)
for study in studies:
    study.non_dominated_sorting()
    study.calculate_hv_history(reference_point)
    study.calculate_hv(reference_point)
    # print(study.sampler_type)
    sum_hv[study.sampler_type].append(study.hv)
print("mean and sd:")
for i in sum_hv.keys():
    print(i)


    print(statistics.mean(sum_hv[i]))
    print(statistics.stdev(sum_hv[i]))

pareto_front_df = pd.DataFrame(columns=('Sampler', 'Seed', 'PCC','FLOPS','Front Rank', 'Pareto optimal'))

# sns.color_palette("flare", as_cmap=True)

for study in studies:
    for rank, trial in study.ranked_fronts:
        pareto = True if rank == 0 else False

        row_dict = {'Sampler': study.sampler_type, 'Seed' : study.seed, 'PCC': 1- trial[0], 'FLOPS': trial[1], 'Front Rank': str(rank+1), 'Pareto optimal':pareto}
        pareto_front_df = pareto_front_df.append(row_dict, ignore_index = True)

plt.figure(figsize=(4, 3))
sns.relplot(data = pareto_front_df, x='PCC', y='FLOPS', alpha = 0.6, row = 'Sampler', col = 'Seed', hue = 'Front Rank', palette = "rocket", style = 'Pareto optimal')

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hv_history_to_df(study):
    df = None
    row_dict = {'sampler_type': [study.sampler_type], 'iteration' : [0], 'hypervolume' : [0.0]}
    df = pd.DataFrame(data = row_dict)
    for index, hv in enumerate(study.hv_history):
        row_dict = {'sampler_type': study.sampler_type, 'iteration' : index + 1, 'hypervolume' : hv}
        df = df.append(row_dict, ignore_index = True)

    return df



hv_history = pd.DataFrame(columns=('sampler_type','iteration', 'hypervolume'))
for study in studies:
    hv_history = hv_history.append(hv_history_to_df(study))

hv_mean_history = hv_history.groupby(['sampler_type','iteration']).mean()

sns.set_theme(style="darkgrid")

plt.figure(figsize =(4,3))
lines =sns.lineplot(
    data = hv_mean_history,
    x = 'iteration',
    y = 'hypervolume',
    hue= 'sampler_type',
    palette = 'bright'

)
lines.set(xlabel="Iterations", ylabel="Mean Hypervolume", )
plt.show()

def get_n_best_SOO(study, n):
    results = sorted(study.objective_list, key = (lambda x : x[0]))

    return results[:n]

def get_faster_trial(study, flops):

    if study.sampler_type == "TPESampler":
        faster_trials = [ i for i in study.objective_list if i[1] < flops]

    else:
        faster_trials = [ i for i in study.objective_list if i[1] <= flops]

    if len(faster_trials) == 0:
        return sorted(study.objective_list, key = (lambda x: x[1]))[0]
    elif len(faster_trials) == 1:
        best = sorted(faster_trials, key = (lambda x : x[0]))[0]
        return best
    else:
        best = sorted(faster_trials, key = (lambda x : x[0]))[0]
    return best



to_beat_dict = defaultdict(list)
answers = defaultdict(list)
n = 1
number_of_seeds = 3
for study in studies:
    if study.sampler_type == "TPESampler":
        to_beat_dict[study.seed] = get_n_best_SOO(study, n)


for i in sampler_names:
    answers[i] = [[0.0, 0.0] ]*n
print(answers)
for study in studies:

    to_beat = to_beat_dict[study.seed]
    for index, best in enumerate(to_beat):
        faster_trial = get_faster_trial(study, best[1])
        print(faster_trial)
        print(index)

        answers[study.sampler_type][index][0] +=  faster_trial[0]
        answers[study.sampler_type][index][1] +=  faster_trial[1]





for key in answers.keys():
    for index, score  in enumerate(answers[key]):
        answers[key][index][0] = score[0] / number_of_seeds
        answers[key][index][1] = score[1] / number_of_seeds

sums = [0.0, 0.0] * n
for key in to_beat_dict.keys():
    lenght = len(to_beat_dict[key])

    for i in to_beat_dict[key]:
        sums[0] += i[0]
        sums[1] += i[1]
    to_beat = [y / 3 for y in sums]


plt.clf()
fig, ax = plt.subplots()
colors = {"RandomSampler": "green", "MOTPESampler": "blue", "NSGAIISampler": "red", "TPESampler": 'brown'}
for i in answers.keys():
    x = 1 - to_beat[0]
    y = to_beat[1]
    u = 1 - answers[i][0][0]  - x
    v =  answers[i][0][1] - y
    print(f"x {x},y {y},u {u},v {v}," )
    ax.arrow( x= x, y=y, dx= u, dy= v, label = i, color = colors[i], length_includes_head=True, head_width = 0.002, head_length = 10000, width = 0.001)
ax.set_ylabel('FLOPS')
ax.set_xlabel('PCC')
plt.legend(loc='lower right')
plt.show()

hypervolume_dict = defaultdict(list)
for study in studies:
    hypervolume_dict[study.sampler_type].append(study.hv)
kruskal = scipy.stats.kruskal(*[ hypervolume_dict[i] for i in hypervolume_dict.keys()])
print(kruskal)

pearson = 0
flops = 0
for trial in all_random_objectives:
    pearson += 1 - trial[0]
    flops += trial[1]

n = len(all_random_objectives)
print(f"expected random value:{pearson/n}, {flops/n}")

def mean_expected_value(name):
    pearson = 0
    flops = 0
    for study in  studies:
        if study.sampler_type == name:
            for trial in study.objective_list:
                pearson += 1 - trial[0]
                flops += trial[1]
    print(f"expected value {name} :{pearson/n}, {flops/n}")

mean_expected_value("MOTPESampler")

mean_expected_value("NSGAIISampler")
mean_expected_value("TPESampler")

PCC = defaultdict(list)

for study in studies:
    for objective in study.objective_list:
        PCC[study.sampler_type].append(1-objective[0])
print(scipy.stats.kruskal(*[PCC[i] for i in PCC.keys()]))
print(scipy.stats.wilcoxon(PCC['TPESampler'],PCC['MOTPESampler'],alternative= 'greater'))
print(scipy.stats.wilcoxon(PCC['TPESampler'],PCC['NSGAIISampler'],alternative= 'greater'))
print(scipy.stats.wilcoxon(PCC['TPESampler'],PCC['RandomSampler'],alternative= 'greater'))
# print(wilcoxon)

Flops = defaultdict(list)

for study in studies:
    for objective in study.objective_list:
        Flops[study.sampler_type].append(objective[1])
print(scipy.stats.kruskal(*[Flops[i] for i in Flops.keys()]))
print(scipy.stats.wilcoxon(Flops['TPESampler'],Flops['MOTPESampler'],alternative= 'greater'))
print(scipy.stats.wilcoxon(Flops['TPESampler'],Flops['NSGAIISampler'],alternative= 'greater'))
print(scipy.stats.wilcoxon(Flops['TPESampler'],Flops['RandomSampler'],alternative= 'greater'))
# print(wilcoxon)

pearson = [1 - i[0] for i in all_random_objectives]
flops = [i[1] for i in all_random_objectives]

spearman = scipy.stats.spearmanr(a=flops, b=pearson, alternative = 'greater')

print(spearman)
