import pygmo
import optuna
import pickle
from collections import defaultdict
import sklearn
import statistics
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


path_prefix = Path("/home/jakob/src/siamese_attention_thesis/studies/")

study_paths = [('RandomSampler', '0_RandomSampler_40'),
               ('MOTPE', '0_MOTPESampler_40'),
               ('NSGAII', '0_NSGAII_40')]


samplers = ["RandomSampler", "MOTPE", "NSGAII"]
pareto_front_dict = defaultdict(lambda :defaultdict(list))
candidate_solution_list = []
hter_corr_inverted_list= []
eval_time_list= []
epoch_time_list= []


def find_nadir_point():
    ''' this function returns a point that is slightly worse than the worst point in all trials'''
    pass

def find_reference_front():
    '''This function returns the global pareto optimal front'''
    pass

def aggregate_studies():
    ''' here we aggregate all the trials from all the studies so we can identify some properties'''
    pass
for sampler, path in study_paths:
    study_path = path_prefix / (path + '.pkl')
    with open(study_path, 'rb') as handle:
        study = pickle.load(handle)
        trials = study.trials
        best_trials = study.best_trials

        for trial in trials:
            # print(trial.values)
            hter_corr, flops = trial.values
            hter_corr_inverted = hter_corr * -1 + 1

            candidate_solution_list.append((
                hter_corr_inverted, flops))

        normalizer = MinMaxScaler().fit(candidate_solution_list)
        scaled_candidates = normalizer.transform(candidate_solution_list)

        for best_trial in best_trials:
            # print(best_trial.values)
            hter_corr, flops = best_trial.values
            hter_corr_inverted = hter_corr * -1 + 1
            objectives =  [(hter_corr_inverted, flops)]
            # print(objectives)
            scaled_objectives = tuple(normalizer.transform(objectives)[0])
            pareto_front_dict[sampler][int(path[0])].append(scaled_objectives)

# print(pareto_front_dict)

fronts, _, _, _ = pygmo.fast_non_dominated_sorting(scaled_candidates)
hi = pygmo.fast_non_dominated_sorting(scaled_candidates)
reference_front = fronts[1]
nadir_point = pygmo.nadir(scaled_candidates)
nadir_point = [ i + 1  for i in nadir_point]
all_hv = pygmo.hypervolume(scaled_candidates)
ref_point = all_hv.refpoint(1)
nadir_point = (1,1)
# print(nadir_point)

for sampler in samplers:

    front = pareto_front_dict[sampler][0]
    hv = pygmo.hypervolume(front)
    print(sampler)
    print(hv.compute(nadir_point))
