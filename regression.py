'''
Linear regression

Val Alvern Cueco Ligo

Main file for linear regression and model selection.
'''

from sklearn.model_selection import train_test_split
import util
import numpy as np


class DataSet(object):
    '''
    Class for representing a data set.
    '''

    def __init__(self, dir_path):
        '''
        Class for representing a dataset, performs train/test
        splitting.

        Inputs:
            dir_path: (string) path to the directory that contains the
              file
        '''

        parameters_dict = util.load_json_file(dir_path, "parameters.json")
        self.feature_idx = parameters_dict["feature_idx"]
        self.name = parameters_dict["name"]
        self.target_idx = parameters_dict["target_idx"]
        self.training_fraction = parameters_dict["training_fraction"]
        self.seed = parameters_dict["seed"]
        self.labels, data = util.load_numpy_array(dir_path, "data.csv")

        # do standardization before train_test_split
        if (parameters_dict["standardization"] == "yes"):
            data = self.standardize_features(data)

        self.training_data, self.testing_data = train_test_split(data,
                                                                 train_size=self.training_fraction, test_size=None,
                                                                 random_state=self.seed)

    # data standardization
    def standardize_features(self, data):
        '''
        Standardize features to have mean 0.0 and standard deviation 1.0.
        Inputs:
          data (2D NumPy array of float/int): data to be standardized
        Returns (2D NumPy array of float/int): standardized data
        '''
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        return (data - mu) / sigma


class Model(object):
    '''
    Class for representing a model.
    '''

    def __init__(self, dataset, feature_idx):
        '''
        Construct a data structure to hold the model.
        Inputs:
            dataset: an dataset instance
            feature_idx: a list of the feature indices for the columns (of the
              original data array) used in the model.
        '''

        self.dataset = dataset
        self.feature_idx = feature_idx
        self.target_idx = dataset.target_idx
        self.x_train = \
            util.prepend_ones_column(dataset.training_data[:, self.feature_idx])
        self.y_train = dataset.training_data[:, self.target_idx]
        self.beta = util.linear_regression(self.x_train, self.y_train)
        self.R2 = self.predict_r2(self.x_train, self.y_train)

    def __repr__(self):
        '''
        Format model as a string.
        '''
        labels = np.array(self.dataset.labels)
        b = ""
        for i, feat in enumerate(self.feature_idx):
            b += "+ {:.6f} * {}".\
                format(self.beta[1:][i], labels[feat])

        return "CRIME_TOTALS ~ {:.6f} {}"\
            .format(self.beta[0], b)

    def predict_r2(self, X, y):
        '''
        Calculates R2 for given data from trained model

        Inputs:
            X: feature variables
            y: target variables

        Returns:
            R2 score (float)
        '''
        y_hat = util.apply_beta(self.beta, X)
        y_mean = np.mean(y)
        r2 = 1 - ((np.sum((y - y_hat) ** 2)) /
                       (np.sum((y - y_mean) ** 2)))

        return r2

def compute_single_var_models(dataset):
    '''
    Computes all the single-variable models for a dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        List of Model objects, each representing a single-variable model
    '''

    return [Model(dataset, [feature]) for feature in dataset.feature_idx]

def compute_all_vars_model(dataset):
    '''
    Computes a model that uses all the feature variables in the dataset

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object that uses all the feature variables
    '''

    return Model(dataset, dataset.feature_idx)

def compute_best_pair(dataset):
    '''
    Find the bivariate model with the best R2 value

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A Model object for the best bivariate model
    '''

    features = dataset.feature_idx
    r2 = 0
    for ind_1 in features:
        for ind_2 in features[ind_1 + 1:]:
            bi = []
            bi.extend([ind_1, ind_2])
            model = Model(dataset, bi)
            x_train, y_train = (model.x_train, model.y_train)
            model.predict_r2(x_train, y_train)
            if model.R2 > r2:
                r2 = model.R2
                best = model

    return best

def forward_selection(dataset):
    '''
    Given a dataset with P feature variables, uses forward selection to
    select models for every value of K between 1 and P.

    Inputs:
        dataset: (DataSet object) a dataset

    Returns:
        A list (of length P) of Model objects. The first element is the
        model where K=1, the second element is the model where K=2, and so on.
    '''

    features = dataset.feature_idx
    refer = features[:]
    models = []
    feat = []
    for k in range(len(features)):
        best_r2 = 0
        for feature in refer:
            a = feat + [feature]
            model = Model(dataset, a)
            x_train, y_train = (model.x_train, model.y_train)
            model.predict_r2(x_train, y_train)
            if model.R2 > best_r2:
                best_r2 = model.R2
                best_feat = feature
                selected = model
        feat.append(best_feat)
        models.append(selected)
        refer.remove(best_feat)

    return models

def validate_model(dataset, model):
    '''
    Given a dataset and a model trained on the training data,
    compute the R2 of applying that model to the testing data.

    Inputs:
        dataset: (DataSet object) a dataset
        model: (Model object) A model that must have been trained
           on the dataset's training data.

    Returns:
        (float) An R2 value
    '''
    test = dataset.testing_data
    x_test = util.prepend_ones_column(test[:, model.feature_idx])
    y_test = test[:, model.target_idx]
    r2 = model.predict_r2(x_test, y_test)

    return r2
