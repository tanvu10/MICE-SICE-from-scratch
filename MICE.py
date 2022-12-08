import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
import statistics as st
import time



class MICE:
    def __init__(self, data, binary_set, bin_algorithm, cont_algorithm, threshold):
        self.X = data
        self.binary_set = binary_set
        self.bin_algorithm = bin_algorithm
        self.cont_algorithm = cont_algorithm
        self.threshold = threshold

        self.nan_dic = {i: [] for i in range(self.X.shape[1])}
        self.nan_dic = self.nan_missing_create(self.nan_dic, type='nan_dic')

        self.fitted_X = deepcopy(self.X)


    def nan_missing_create(self, target_dic, type='nan_dic'):
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[0]):
                if np.isnan(self.X[j, i]):
                    if type == 'nan_dic':
                        target_dic[i].append(j)
                    else:
                        # 2d list
                        # 2nd element: prediction, 3rd element: error
                        target_dic[i].append([j, 0, 0])
        return target_dic

        # self.nan_dic[i].append(j)

    # use for first iteration
    def place_holder(self, X_sample, option='MEAN'):
        n_col = X_sample.shape[1]
        for col in range(n_col):
            if option == 'MEAN' and col not in self.binary_set:
                mean = np.nanmean(X_sample[:, col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = mean

            elif option == 'MEDIAN' and col not in self.binary_set:
                med = np.nanmedian(X_sample[:, col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = med

            else:
                mode = st.mode(X_sample[:, col])
                for i in self.nan_dic[col]:
                    X_sample[i, col] = mode

        return X_sample

    def LR_one_iter_fit(self, current_dataset, it_dic):
        # it_dic: # 2d dic second element in this dic[list] to store value of prediction
        max_error = 0
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

            # reg = LinearRegression().fit(X_, y)

            # take original X without col_i
            sub_X = np.delete(copy_dataset, col, 1)

            for i in range(len(self.nan_dic[col])):
                missing_row = self.nan_dic[col][i]
                prediction_value = reg.predict(sub_X[missing_row, :].reshape(-1, 1).T)

                # replace to final_ori_data not copy
                final_ori_dataset[missing_row, col] = prediction_value

                pre_prediction_value = it_dic[col][i][1]
                if col not in self.binary_set:
                    max_error = max(max_error, abs(prediction_value - pre_prediction_value))

                # save this prediction, error value to this iteration_dic:
                it_dic[col][i][1] = prediction_value
                it_dic[col][i][2] = abs(prediction_value - pre_prediction_value)

        return final_ori_dataset, it_dic, max_error

    def MICE_fit(self):
        max_error = 10e9
        # just set for the initial time
        self.fitted_X = self.place_holder(X_sample=self.fitted_X)
        i = 0
        MICE_iter_dic = {j: [] for j in range(self.X.shape[1])}
        # 2nd element: prediction, 3rd element: error
        MICE_iter_dic = self.nan_missing_create(MICE_iter_dic, type='other')
        dif_error = 10e9
        while dif_error > self.threshold:
            print(f'iteration {i + 1}')
            self.fitted_X, MICE_iter_dic, max_it_error = self.LR_one_iter_fit(self.fitted_X, MICE_iter_dic)
            print(max_it_error)
            dif_error = abs(max_it_error - max_error)
            print(dif_error)
            max_error = max_it_error
            i += 1

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
    mice_model = MICE(data=missing_data_set, binary_set=bin_set
                      , bin_algorithm='random_forest', cont_algorithm='linear_regression', threshold=10**(-4))
    # sice_model = SICE(data=missing_data_set, n_iteration=10, binary_set=bin_set
    #                   , bin_algorithm='log_regression', cont_algorithm='lasso_regression')
    t0 = time.time()
    mice_model.MICE_fit()
    t1 = time.time() - t0
    print("Time elapsed: ", t1)
    # sice_model.SICE_result()
    print(pd.DataFrame(mice_model.fitted_X))
    error = 0
    n_col = full_data_set.shape[1]
    for col in range(n_col):
        for row in mice_model.nan_dic[col]:
            error += (mice_model.fitted_X[row, col] - full_data_set[row, col]) ** 2
    print(error)