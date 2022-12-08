import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
import statistics as st
import time
class SICE:
    def __init__(self, data, n_iteration, binary_set, bin_algorithm, cont_algorithm):
        self.X = data
        self.iteration = n_iteration
        self.nan_dic = {i: [] for i in range(self.X.shape[1])}
        self.nan_dic = self.nan_missing_create(self.nan_dic, type='nan_dic')

        self.result_dic = {} # (row, col) = []
        self.result_dic = self.nan_missing_create(self.result_dic, type='result')

        self.fitted_X = deepcopy(self.X)
        self.SICE_value_dic = {}

        self.binary_set = binary_set
        self.bin_algorithm = bin_algorithm
        self.cont_algorithm = cont_algorithm

    def nan_missing_create(self, target_dic, type='nan_dic'):
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[0]):
                if np.isnan(self.X[j, i]):
                    if type == 'nan_dic':
                        target_dic[i].append(j)
                    elif type == 'result':
                        target_dic[(j, i)] = [] # row, col
                    else:
                        # 2d list
                        # second element in this list to store error or value of prediction
                        target_dic[i].append([j, 0])
        return target_dic

                    # self.nan_dic[i].append(j)

    # use for first iteration
    def place_holder(self, X_sample, option='MEAN'):
        n_col = X_sample.shape[1]
        for col in range(n_col):
            if option == 'MEAN' and col not in self.binary_set:
                mean = np.nanmean(X_sample[:,col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = mean

            elif option == 'MEDIAN' and col not in self.binary_set:
                med = np.nanmedian(X_sample[:, col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = med

            else: # auto MODE for binary data
                mode = st.mode(X_sample[:, col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = mode

        return X_sample

    def LR_one_iter_fit(self, current_dataset, it_dic):
        # it_dic: # 2d dic second element in this dic[list] to store value of prediction

        # make sure there is no nan in this current_dataset
        # copy dataset: taking out value for further prediction
        copy_dataset = deepcopy(current_dataset)

        # final_ori_dataset: replace predicted value to this dataset
        final_ori_dataset = deepcopy(current_dataset)
        n_col = copy_dataset.shape[1]

        for col in range(n_col):

            # temp remove all missing data of col_i
            sub_dataset = np.delete(current_dataset, self.nan_dic[col], 0)
            y = sub_dataset[:, col]
            X_ = np.delete(sub_dataset, col, 1)

            if col in self.binary_set:
                if self.bin_algorithm == 'random_forest':
                    # creating a RF classifier
                    reg = RandomForestClassifier(n_estimators=100)
                    reg.fit(X_, y)
                else:
                    reg = LogisticRegression().fit(X_, y)
            else:
                if self.cont_algorithm == 'linear_regression':
                    reg = LinearRegression().fit(X_, y)
                else:
                    reg = Lasso().fit(X_, y)

            # take original X without col_i
            sub_X = np.delete(copy_dataset, col, 1)

            for i in range(len(self.nan_dic[col])):
                missing_row = self.nan_dic[col][i]
                prediction_value = reg.predict(sub_X[missing_row, :].reshape(-1, 1).T)

                # print(missing_row, col)
                # replace to final_ori_data not copy
                final_ori_dataset[missing_row, col] = prediction_value

                # save this prediction value to this iteration_dic:
                it_dic[col][i][1] = prediction_value

        return final_ori_dataset, it_dic

    def SICE_fit(self):
        # just set for the initial time
        self.fitted_X = self.place_holder(X_sample=self.fitted_X)
        sample_MICE_iter_dic = {j: [] for j in range(self.X.shape[1])}
        sample_MICE_iter_dic = self.nan_missing_create(sample_MICE_iter_dic, type='other')

        for i in range(self.iteration):
            print(f'iteration {i+1}')
            MICE_iter_dic = deepcopy(sample_MICE_iter_dic)
            self.fitted_X, MICE_iter_dic = self.LR_one_iter_fit(self.fitted_X, MICE_iter_dic)
            self.SICE_value_dic[i] = MICE_iter_dic

    def SICE_result(self):
        n_col = self.fitted_X.shape[1]
        for i in range(self.iteration): # self.iteration is small (guarantee in O(n^2))
            iter_dic = self.SICE_value_dic[i]

            for col in range(n_col):
                for j in range(len(self.nan_dic[col])):
                    row = iter_dic[col][j][0]
                    iter_result = iter_dic[col][j][1]
                    self.result_dic[(row, col)].append(iter_result)

        for row, col in self.result_dic.keys():
            self.fitted_X[row, col] = np.mean(self.result_dic[(row, col)])


def make_binary_set(data):
    bin_set = set()
    new_data = data.dropna()
    for col in range(new_data.shape[1]):
        if np.array_equal(new_data.iloc[:, col], new_data.iloc[:, col].astype(bool)):
            bin_set.add(col)
    return bin_set

if __name__ == '__main__':
    path = '/Users/tanvu10/Downloads/'
    missing_data_file_name = 'sample (1).csv'
    full_data_file_name = 'sample_full (1).csv'

    full_data_set = pd.read_csv(path + full_data_file_name)
    full_data_set = full_data_set.iloc[:, 1:]
    full_data_set = full_data_set.to_numpy()

    missing_data_set = pd.read_csv(path + missing_data_file_name)
    missing_data_set = missing_data_set.iloc[:, 1:]
    bin_set = make_binary_set(missing_data_set)
    missing_data_set = missing_data_set.to_numpy()
    # print(missing_data_set)
    sice_model = SICE(data=missing_data_set, n_iteration=5, binary_set=bin_set
                      ,bin_algorithm='random_forest', cont_algorithm='linear_regression')
    # sice_model = SICE(data=missing_data_set, n_iteration=10, binary_set=bin_set
    #                   , bin_algorithm='log_regression', cont_algorithm='lasso_regression')
    t0 = time.time()
    sice_model.SICE_fit()
    t1 = time.time() - t0
    print("Time elapsed: ", t1)
    sice_model.SICE_result()
    print(pd.DataFrame(sice_model.fitted_X))
    error = 0
    n_col = full_data_set.shape[1]
    for col in range(n_col):
        for row in sice_model.nan_dic[col]:
            error += (sice_model.fitted_X[row, col] - full_data_set[row,col])**2
    print(error)


