from skmultiflow.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.options.file_option import FileOption
from skmultiflow.data.file_stream import FileStream
from AdaptiveRandomForest import AdaptiveRandomForest
from skmultiflow.meta.leverage_bagging import LeverageBagging
""""
import pandas as pd
#读取data文件，指定属性，sep='[\s]*'意义为匹配一个或多个空格，因为原始数据集中数据分割是两个或者多个空格
data=pd.read_table('elecNorm.data',header=None,sep=',')
#生成csv文件
data.to_csv('elec.csv',index=False)
"""


def run_experiment(dataset="elec", pre_train_size=1000, max_instances=10000, batch_size=1, n_wait=500, max_time=1000000000, task_type='classification', show_plot=False,
                                plot_options=['performance']):

    # 1. Create a stream
    opt = FileOption("FILE", "OPT_NAME", dataset+".csv", "CSV", False)
    stream = FileStream('elec1.csv')
    # 2. Prepare for use
    stream.prepare_for_use()
    # 2. Instantiate the HoeffdingTree classifier

    h = [
            HoeffdingTree(),
            AdaptiveRandomForest(nb_features=6, nb_trees=100, predict_method="avg", pretrain_size=pre_train_size,
                                 delta_d=0.001, delta_w=0.01),
            AdaptiveRandomForest(nb_features=6, nb_trees=5, predict_method="avg", pretrain_size=pre_train_size,
                                 delta_d=0.001, delta_w=0.01)
            #AdaptiveRandomForest(nb_features=3, nb_trees=80, predict_method="avg", pretrain_size=pre_train_size,
              #                   delta_d=0.001, delta_w=0.01)
            #AdaptiveRandomForest(m=8, n=25)
         ]
    # 3. Setup the evaluator
    eval1 = EvaluatePrequential(pretrain_size=pre_train_size, output_file='result_'+dataset+'.csv',
                                batch_size=batch_size, n_wait=n_wait, max_time=max_time,  show_plot=show_plot,
                                )
    # 4. Run
    eval1.evaluate(stream=stream,model=HoeffdingTree())


run_experiment(dataset="elec", pre_train_size=2000, max_instances=10000, batch_size=3, n_wait=500, max_time=1000000000, task_type='classification', show_plot=True,
                                plot_options=['performance'])