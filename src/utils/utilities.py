import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def emotion_codes():
    emotion_codes = {
        'W': 'Ärger (Wut)',
        'L': 'Langeweile',
        'E': 'Ekel',
        'A': 'Angst',
        'F': 'Freude',
        'T': 'Trauer',
        'N': 'Neutral'
    }
    return emotion_codes


def plot_confusionmatrix(clf, y_test, y_pred, labels, cmap=plt.cm.Blues, title=''):
    """Utility to plot confusion matrix from Classifier
    clf - sklearn trained classifier
    y_test - test data. They are preprocessed, e.g. run through a label encoder
    y_pred - they come from the model
    labels - label names(len N Classes)
    cmap - color map)
    """

    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=cmap)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, rotation=90)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def nzvKJ(df, varlist, uniqueCut=10, freqCut=float(95) / 5):
    """
    Purpose: remove variables with near zero variance, Kuhn and Johnson
    alternative: Thresholding, see here http://scikit-learn.org/stable/modules/feature_selection.html

    :param df: pandas df to process
    :varlist: list of variables to process

    Usage:

        get the list of variables to drop with
        nearZeroVar =  nzvKJ(X_train,X_train.columns)
        nearZeroVar =  nzvKJ(X_train,X_train.columns,freqCut = 20.)
        varlist2drop = nearZeroVar[nearZeroVar['nzv']].index.tolist()

        drop with:
        df.drop(varlist2drop, axis=1, inplace=True)

        show it:
        sorted(nearZeroVar[nearZeroVar['nzv'] == True].variance)
        mean(nearZeroVar[nearZeroVar['nzv'] == False].variance)

    http://stats.stackexchange.com/questions/145602/justification-for-feature-selection-by-removing-predictors-with-near-zero-varian

    [Near-zero variance means

    - that the] fraction of unique values over the sample size is low (say 10%) [...] [and the]
    - ratio of the frequency of the most prevalent value to the frequency of
    the second most prevalent value is large (say around 20).

    If both of these criteria are true and the model in question is susceptible to this type of predictor,
    it may be advantageous to remove the variable from the model.

    The R mixOmics package uses the cutoffs freqCut = 95/5 and uniqueCut = 10)

    Links:
    [1] Kuhn, M., & Johnson, K. (2013). Applied predictive modeling, New York, NY: Springer.
    [2] mixOmics Package reference - http://www.inside-r.org/packages/cran/mixOmics/docs/nearZeroVar

    """

    N = df[varlist].shape[0]
    """ fraction of unique values over the sample size is low (say 10%) """

    def fracUnique(x):
        return (100 * float(len(x.unique())) / N) < uniqueCut

    """ the ratio of the frequency of the most prevalent value to the frequency of the second most prevalent value
    is large """

    def largeRatio(x):
        valuecounts = x.value_counts()
        if len(valuecounts) < 2:
            return True
        bc = sorted(
            valuecounts, reverse=True)[0:2]
        return (float(bc[0]) / bc[1]) > freqCut

    dfUse = pd.DataFrame(
        df[varlist].apply(fracUnique, axis=0), columns=['Fracunique'])
    dfUse["largeRatio"] = df[varlist].apply(largeRatio, axis=0)
    dfUse['nzv'] = dfUse.all(axis=1)
    dfUse["variance"] = df[varlist].var()
    dfUse["dtype"] = df[varlist].dtypes
    return dfUse
