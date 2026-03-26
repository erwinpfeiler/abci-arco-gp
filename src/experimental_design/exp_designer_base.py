import time
from collections import namedtuple
from typing import Dict, Tuple, Set

import torch.distributed.rpc as rpc
import torch.optim

from src.experimental_design.optimization import random_search, grid_search, gp_ucb

Design = namedtuple('Design', ['interventions', 'info_gain'])

import time
import torch
import cProfile
import pstats
import io


class ExpDesignerBase:
    def __init__(self, intervention_bounds: Dict[str, Tuple[float, float]], opt_strategy: str = 'gp-ucb',
                 distributed=False):
        self.worker_id = rpc.get_worker_info().id if distributed else 0
        self.intervention_bounds = intervention_bounds
        if opt_strategy not in {'gp-ucb', 'random', 'grid'}:
            print('Invalid optimization strategy ' + opt_strategy + '. Doing Bayesian optimization instead.')
            opt_strategy = 'gp-ucb'
        self.opt_strategy = opt_strategy
        self.utility = None

    def init_design_process(self, args: dict):
        raise NotImplementedError

    def run_distributed(self, experimenter_rref, args: dict):
        experimenter_rref.rpc_sync().report_status(self.worker_id, 'Initializing design process...')
        self.init_design_process(args)
        experimenter_rref.rpc_sync().report_status(self.worker_id, 'Finished initializing design process...')

        target_node = experimenter_rref.rpc_sync().get_target(self.worker_id)
        while target_node:
            design = self.design_experiment(target_node)
            experimenter_rref.rpc_sync().report_design(self.worker_id, target_node, design)
            target_node = experimenter_rref.rpc_sync().get_target(self.worker_id)

    def design_experiment(self, target_node: str):

        # if no target is given report info gain of observational sample
        if target_node == 'OBSERVATIONAL':
            try:
                score = self.utility({})
            except Exception as e:
                print(f'Exception occured in ExperimentDesigner.design_experiment() when the score for the '
                      f'observational target:')
                print(e)
                score = torch.tensor(0.)
            return Design({}, score)

        # otherwise, design experiment for target node
        bounds = torch.Tensor(self.intervention_bounds[target_node]).view(2, 1)
        if self.opt_strategy == 'random':
            target_value, score = random_search(lambda x: self.utility({target_node: x}), bounds)
        elif self.opt_strategy == 'grid':
            target_value, score = grid_search(lambda x: self.utility({target_node: x}), bounds)
        else:
            target_value, score = gp_ucb(lambda x: self.utility({target_node: x}), bounds)
            

        return Design({target_node: target_value}, score)
    

    def get_best_experiment(self, target_nodes: Set[str]):
        best_intervention = {}
        print(f"Starting initial utility eval...", flush=True)
        t0 = time.time()
        best_score = self.utility({})
        t1 = time.time()
        print(f"Finished initial utility eval", flush=True)
        print(f"Time needed = {t1-t0:.4f} s\n", flush=True)
        print(f'Expected information gain for observational sample is {best_score}.', flush=True)
        for target_node in target_nodes:
            print(f'Start experiment design for node {target_node} at {time.strftime("%H:%M:%S")}', flush=True)
            t0 = time.time()
            design = self.design_experiment(target_node)
            t1 = time.time()
            print(f'Expected information gain for {design.interventions} is {design.info_gain}.', flush=True)
            print(f"Time needed = {t1-t0:.4f} s\n", flush=True)
            if design.info_gain > best_score:
                best_score = design.info_gain
                best_intervention = design.interventions

        return best_intervention



"""
First call 5 nodes
------ POSTERIOR EXPECTATION PROFILE ------
Total time:            0.0919s
Parent set generation: 0.0017s
Func eval (node_mll):  0.0836s
Stack time:            0.0009s
logsumexp time:        0.0029s
-------------------------------------------


========== GRAPH INFO GAIN PROFILE ==========
Total time:              10.3964s
Total loop time:         10.3963s
Graph building:          0.0202s
Topo init:               0.0067s
Simulation:              1.0725s
Cache clearing:          0.0015s
Inner expectation:       9.2873s
Outer MLL:               0.0062s
=============================================

Expected information gain for {'X3': tensor(0.8929)} is -1.051398515701294.
         812305385 function calls (778462886 primitive calls) in 769.544 seconds

   Ordered by: cumulative time
   List reduced from 635 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       76    0.001    0.000  761.846   10.024 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:61(_graph_info_gain)
       76    0.324    0.004  761.843   10.024 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/autograd/grad_mode.py:273(__exit__)
     7600   24.466    0.003  684.673    0.090 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:191(graph_posterior_expectation_factorising)
  3800000    2.234    0.000  615.491    0.000 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:125(pred_log_node)
  3838000    4.125    0.000  613.426    0.000 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/shared_data_gp_model.py:237(node_mll)
   122000    0.646    0.000  583.782    0.005 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:620(mll)
   135725    0.511    0.000  561.296    0.004 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:601(get_ydist)
   135725    1.352    0.000  536.954    0.004 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:588(get_fdist)
   138275    4.505    0.000  529.821    0.004 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_gp.py:264(__call__)
974125/428225    3.331    0.000  287.076    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/utils/memoize.py:54(g)
   135775    2.398    0.000  258.680    0.002 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:311(exact_prediction)
1370250/819750    3.875    0.000  256.791    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/module.py:81(__call__)
   271450    5.395    0.000  215.430    0.001 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:452(forward)
   271500    0.272    0.000  160.679    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:408(to_dense)
274200/274150    0.884    0.000  160.403    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:22(wrapped)
   274100    2.453    0.000  158.420    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:342(evaluate_kernel)
   271550    1.131    0.000  147.053    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:2295(solve)
   135775    2.683    0.000  125.507    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:328(exact_predictive_mean)
   135775    0.176    0.000  122.297    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:253(mean_cache)
   135775    4.845    0.000  121.682    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:368(exact_predictive_covar)
   135775    1.856    0.000  120.846    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:257(_mean_cache)
299000/285275    0.922    0.000  117.137    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/autograd/function.py:559(apply)
299000/285275    3.892    0.000  113.760    0.000 {built-in method apply}
1243050/409775    4.649    0.000  110.624    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/utils/memoize.py:54(g)
   409775    0.369    0.000  109.207    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:1301(cholesky)
   271550    1.804    0.000  108.595    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/functions/_solve.py:25(forward)
   409775    2.759    0.000  104.730    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:500(_cholesky)
   670050    1.828    0.000  100.519    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/added_diag_linear_operator.py:208(evaluate_kernel)
  6783855   48.652    0.000   93.432    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/functional.py:79(broadcast_shapes)
   545550    1.071    0.000   93.116    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/likelihood.py:70(__call__)
  2292975    6.097    0.000   91.642    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/added_diag_linear_operator.py:37(__init__)
   545550    0.748    0.000   91.586    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/gaussian_likelihood.py:173(marginal)
   545550    2.025    0.000   90.837    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/gaussian_likelihood.py:114(marginal)
   271550    1.221    0.000   88.349    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/functions/_solve.py:9(_solve)
18405900/14113075    7.843    0.000   86.650    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:2291(shape)
  2292975   13.128    0.000   83.967    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/sum_linear_operator.py:18(__init__)
   124500    2.462    0.000   82.403    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/distributions/multivariate_normal.py:222(log_prob)
   548150    2.800    0.000   80.605    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/kernels/kernel.py:459(__call__)
     7600    0.016    0.000   74.349    0.010 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:51(_simulate_experiment)
     7600    0.652    0.000   74.333    0.010 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/shared_data_gp_model.py:322(sample)


Second call 5 nodes:
        906386144 function calls (868363925 primitive calls) in 839.266 seconds

   Ordered by: cumulative time
   List reduced from 635 to 40 due to restriction <40>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       76    0.001    0.000  831.754   10.944 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:61(_graph_info_gain)
       76    0.358    0.005  831.751   10.944 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/autograd/grad_mode.py:273(__exit__)
     7600   25.044    0.003  732.481    0.096 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:191(graph_posterior_expectation_factorising)
  3800000    2.295    0.000  661.684    0.000 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:125(pred_log_node)
  3838000    4.250    0.000  659.566    0.000 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/shared_data_gp_model.py:237(node_mll)
   134200    0.696    0.000  629.195    0.005 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:620(mll)
   152805    0.562    0.000  620.980    0.004 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:601(get_ydist)
   152805    1.488    0.000  594.368    0.004 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:588(get_fdist)
   155355    4.944    0.000  586.404    0.004 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_gp.py:264(__call__)
1093685/479465    3.657    0.000  317.904    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/utils/memoize.py:54(g)
1541050/922230    4.286    0.000  285.783    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/module.py:81(__call__)
   152855    2.666    0.000  284.134    0.002 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:311(exact_prediction)
   305610    6.011    0.000  240.784    0.001 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:452(forward)
   305660    0.299    0.000  179.543    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:408(to_dense)
308360/308310    0.965    0.000  179.027    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:22(wrapped)
   308260    2.711    0.000  176.873    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/lazy/lazy_evaluated_kernel_tensor.py:342(evaluate_kernel)
   305710    1.248    0.000  161.690    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:2295(solve)
   152855    3.020    0.000  137.795    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:328(exact_predictive_mean)
   152855    0.189    0.000  134.186    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:253(mean_cache)
   152855    5.393    0.000  133.679    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:368(exact_predictive_covar)
   152855    2.070    0.000  132.570    0.001 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/models/exact_prediction_strategies.py:257(_mean_cache)
342920/324315    1.056    0.000  129.648    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/autograd/function.py:559(apply)
342920/324315    4.394    0.000  125.896    0.000 {built-in method apply}
1401650/461015    5.124    0.000  122.014    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/utils/memoize.py:54(g)
   461015    0.401    0.000  120.078    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:1301(cholesky)
   305710    1.996    0.000  119.818    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/functions/_solve.py:25(forward)
   461015    3.027    0.000  115.223    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:500(_cholesky)
   750570    2.011    0.000  108.870    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/added_diag_linear_operator.py:208(evaluate_kernel)
   613870    1.164    0.000  101.809    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/likelihood.py:70(__call__)
  7654935   53.029    0.000  101.412    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/torch/functional.py:79(broadcast_shapes)
   613870    0.810    0.000  100.149    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/gaussian_likelihood.py:173(marginal)
  2573575    6.645    0.000   99.611    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/added_diag_linear_operator.py:37(__init__)
   613870    2.229    0.000   99.338    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/likelihoods/gaussian_likelihood.py:114(marginal)
   305710    1.357    0.000   97.675    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/functions/_solve.py:9(_solve)
     7600    0.017    0.000   96.175    0.013 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/experimental_design/exp_designer_abci_arco_gp.py:51(_simulate_experiment)
     7600    0.684    0.000   96.158    0.013 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/shared_data_gp_model.py:322(sample)
20660460/15855235    8.613    0.000   94.401    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/_linear_operator.py:2291(shape)
   616470    3.080    0.000   91.659    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/gpytorch/kernels/kernel.py:459(__call__)
  2573575   14.477    0.000   91.273    0.000 /ceph/home/TUG/epfeiler-tug/.conda/envs/abci-arco-gp/lib/python3.12/site-packages/linear_operator/operators/sum_linear_operator.py:18(__init__)
    18605    0.073    0.000   90.962    0.005 /ceph/home/TUG/epfeiler-tug/abci-test-arcogp-graph-info-20260302_202918/src/mechanism_models/mechanisms.py:613(sample)


/ceph/home/TUG/epfeiler-tug/abci-arco-gp/slurm-logs/BarabasiAlbert/5_nodes_debug/20260302_212520_test-arcogp-graph-info
contains chatgpt caching (did not help)

/ceph/home/TUG/epfeiler-tug/abci-arco-gp/slurm-logs/BarabasiAlbert/5_nodes_debug/20260302_222259_test-arcogp-graph-info
is without any caching and should be slower to show that my original caching was working and quantify the exact performance
     """