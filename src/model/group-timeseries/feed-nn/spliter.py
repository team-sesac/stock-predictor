from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import numpy as np
import pandas as pd

class PurgedGroupTimeSeriesSplit(_BaseKFold):
    
    @_deprecate_positional_args
    def __init__(self,
                    n_splits=5,
                    *,
                    max_train_group_size=np.inf,
                    max_test_group_size=np.inf,
                    group_gap=None,
                    verbose=False):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                    " the number of groups={1}").format(n_folds,
                                                        n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                        np.concatenate((train_array,
                                                        train_array_tmp)),
                                        axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                                np.concatenate((test_array,
                                                                test_array_tmp)),
                                        axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]
            
if __name__ == "__main__":
    data = pd.read_csv("data/temp/kaggle_train.csv")
    # data = data[data["date_id"] < 478]
    data = data.fillna(method="ffill").fillna(0)
    data = data.drop(labels=["row_id"], axis=1)
    data['Y'] = data.loc[:, ["target"]]
    data = data.drop(labels=["target"], axis=1)
    
    
    spliter = PurgedGroupTimeSeriesSplit()
    # spliter.split(X=data[:, :-1], y=data[:, [-1]], groups=)