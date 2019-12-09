#!/usr/bin/env python
# coding: utf-8
# letter	emotion (english)	letter	emotion (german)
# A	anger	W	Ärger (Wut)
# B	boredom	L	Langeweile
# D	disgust	E	Ekel
# F	anxiety/fear	A	Angst
# H	happiness	F	Freude
# S	sadness	T	Trauer
# N = neutral version
# W L E A F T N
# ### Resampling Training Data


import os
# import dotenv
import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from collections import Counter

from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from sklearn import preprocessing, svm  # noqa
from sklearn.datasets import make_classification  # noqa
from sklearn.metrics import (accuracy_score, classification_report,  # noqa
                             confusion_matrix, f1_score, make_scorer,
                             precision_recall_fscore_support)
from sklearn.mixture import GaussianMixture as GMM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder  # noqa
from sklearn.externals import joblib

project_dir = os.path.dirname(os.path.dirname(os.path.abspath('')))  # might be __file__
print(project_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))
from utils.utilities import nzvKJ, plot_confusionmatrix, emotion_codes  # noqa

from features.custom_transformers import (ColumnExtractor, DateDiffer, # noqa
                                          DateFormatter, DFFeatureUnion,
                                          DFImputer, DFRobustScaler,
                                          DummyTransformer, Log1pTransformer,
                                          MultiEncoder, ZeroFillTransformer)


def read_data():
    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    # dotenv_path = os.path.join(project_dir, '.env')
    # dotenv.load_dotenv(dotenv_path)

    feature_file = os.path.join(project_dir, "data/processed/features.csv")
    filename_file = os.path.join(project_dir, "data/processed/filenames.csv")
    # print(feature_file)

    df = pd.read_csv(feature_file, sep=';')
    filenames = pd.read_csv(filename_file, sep=';')
    y = pd.DataFrame(filenames['filename'].apply(lambda x: x[5]))
    y.columns = ['label']
    speaker = pd.DataFrame(filenames['filename'].apply(lambda x: x[0:2]))
    speaker.columns = ['speaker']

    # print(df.shape)
    # print(df.head())
    wav_path = os.path.join(project_dir, "data/raw/wav")
    wav_files = os.listdir(wav_path)  # noqa
    # print('df shape / y shape / N Wav files {}/{}/{}'.format(df.shape[0], y.shape[0], len(wav_files)))

    return df, y, speaker


def get_feature_names(df):
    """
    Basic Feature Selection

     Only Kuhn Johnson nonzero variabce implemented


     TODO: Improve
     http://scikit-learn.org/stable/modules/feature_selection.html
    """

    """ return feature names that will be selected - and passed to the Pipeline's col extractor """
    dropvars = ['filename', 'name', 'frameTime']
    nearZeroVar = nzvKJ(df, df.columns)
    # nearZeroVar =  nzvKJ(X_train,X_train.columns,freqCut = 20.)
    varlist2drop = nearZeroVar[nearZeroVar['nzv']].index.tolist()
    vars2remove = list(set(varlist2drop + dropvars))
    vars2remove

    NUM_FEATS = list(df.select_dtypes(include=['float64', 'int64']).columns)
    NUM_FEATS = [x for x in NUM_FEATS if x not in vars2remove]
    CAT_FEATS = list(df.select_dtypes(include=['object']).columns)
    CAT_FEATS = [x for x in CAT_FEATS if x not in vars2remove]
    CAT_FEATS
    return NUM_FEATS, CAT_FEATS


def do_train_test_filters(speaker):
    """ return the boolean mask that can be used to do the train test split on the data:
     df_f = df_train[filter]
    """
    random_state = 42
    train_size = 2./3
    speaker_values = list(np.unique(speaker.values))
    # n_speakers = len(speaker_values)  # noqa
    train, test = train_test_split(speaker_values, random_state=random_state, train_size=train_size)
    # print('There should be approx 2/3 of the speakers in train: {}'.format(len(train) / n_speakers))
    train_filter = speaker['speaker'].apply(lambda x: x in train)
    test_filter = speaker['speaker'].apply(lambda x: x in test)
    # print(speaker.shape[0])
    # print(pd.crosstab(train_filter, test_filter))
    return train_filter, test_filter

# y_pred = gmm_clf.predict(X_test_t)
# # classification_report(y_test_enc, y_pred, output_dict=True)
# print(classification_report(y_test_enc, y_pred, output_dict=False))

# #  SVM
# svm_clf = svm.LinearSVC()
# svm_clf.fit(X_train_t_res, y_train_t_res)
# # Alternative:  class_weight='balanced',
# # In[16]:

# X_test_t = pipeline.transform(df_test)
# y_pred = svm_clf.predict(X_test_t)

# print(classification_report(y_test_enc, y_pred))
# classification_report(y_test_enc, y_pred, output_dict=True)

# np.set_printoptions(precision=2)
# print(confusion_matrix(y_test_enc, y_pred))
# label_encoder.classes_

# labels = [emotion_codes[x] for x in label_encoder.classes_]
# plot_confusionmatrix(svm_clf, y_test_enc, y_pred, labels, cmap=plt.cm.Blues)


def get_data():
    # emotion_codes = emotion_codes()
    df, y, speaker = read_data()
    # print(y['label'].value_counts())
    train_filter, test_filter = do_train_test_filters(speaker)

    df_train = df[train_filter]
    df_test = df[test_filter]
    y_train = y[train_filter]
    y_test = y[test_filter]
    # print(df_train.shape[0], df_test.shape[0], y_train.shape[0], y_test.shape[0])
    return df, df_train, df_test, y_train, y_test


class TheEstimator(object):

    def __init__(self):

        df, df_train, df_test, y_train, y_test = get_data()
        NUM_FEATS, CAT_FEATS = get_feature_names(df)
        # Preprocessing with a Pipeline that uses Pandas Capable Processors

        self.pipeline = Pipeline([
            ('features', DFFeatureUnion([
                ('categoricals', Pipeline([
                    ('extract', ColumnExtractor(CAT_FEATS)),
                    ('dummy', DummyTransformer())
                ])),
                ('numerics', Pipeline([
                    ('extract', ColumnExtractor(NUM_FEATS)),
                    ('zero_fill', ZeroFillTransformer())
                ]))
            ])),
            ('std_scaler', preprocessing.StandardScaler())
        ])
        # note ('scale', DFRobustScaler()) is much worse

        self.pipeline.fit(df_train)
        X_train_t = self.pipeline.transform(df_train)
        # X_test_t = self.pipeline.transform(df_test)

        self.label_encoder = LabelEncoder()
        y_train_t = self.label_encoder.fit_transform(np.array(y_train.values))
        # y_test_enc = label_encoder.fit_transform(y_test)
        # print(pd.Series(y_train_t).value_counts())
        # label_encoder.classes_

        sm = SMOTE(random_state=42)
        self.X_train_t_res, self.y_train_t_res = sm.fit_resample(X_train_t, y_train_t)
        # print('Resampled dataset shape %s' % Counter(y_train_t_res))
        # print("resampled shape", X_train_t_res.shape)
        # print("original shape", X_train_t.shape)

    def fit_and_save(self, X, y=None):
        if y is None:
            self.clf.fit(self.X_train_t_res)
        else:
            self.clf.fit(self.X_train_t_res, self.y_train_t_res)

        model_dir = os.path.join(project_dir, 'models')
        clf_outname = os.path.join(model_dir, 'clf_'+self.clf_name+'.pkl')
        # print('dumping results')
        # print('classifier: ', clf_outname)
        joblib.dump(self.clf, clf_outname)

        pipeline_outname = os.path.join(model_dir, 'prepro_pipeline_'+self.clf_name+'.pkl')
        # print('pipeline: ', pipeline_outname)
        joblib.dump(self.pipeline, pipeline_outname)

        label_enc_outname = os.path.join(model_dir, 'label_encoder_'+self.clf_name+'.pkl')
        print('y encoder: ', label_enc_outname)
        joblib.dump(self.label_encoder, label_enc_outname)

    def set_estimator(self, clf, clf_name):
        self.clf = clf
        self.clf_name = clf_name

# def do_train(clf, clf_name):

if __name__ == '__main__':
    n_classes = len(emotion_codes().keys())
    es = TheEstimator()

    gmm_clf = GMM(n_components=n_classes,
                  covariance_type='spherical',
                  n_init=5,
                  means_init=pd.DataFrame(es.X_train_t_res).groupby(es.y_train_t_res).mean().values)

    es.set_estimator(gmm_clf, 'GMM_basic')
    print("Fitting and saving GMM")
    es.fit_and_save(es.X_train_t_res)
    
    clf = svm.LinearSVC()
    es.set_estimator(clf, 'svm_basic')
    print('fitting and saving svm')
    es.fit_and_save(es.X_train_t_res, es.y_train_t_res)


    # gmm_clf = GMM(n_components=n_classes, covariance_type='full', n_init=5, means_init=pd.DataFrame(X_train_t_res).groupby(y_train_t_res).mean().values)
    # gmm_clf = GMM(n_components=500, covariance_type='spherical', n_init=50)
    # gmm_clf.fit(X_train_t_res, y_train_t_res) 
    # gmm_clf.fit(X_train_t_res) 

    
    # clf = GMM(n_components=n_classes, covariance_type='spherical')
    # do_train(clf, 'GMM_basic')
    # clf = svm.LinearSVC()
    # do_train(clf, 'svm_basic')
