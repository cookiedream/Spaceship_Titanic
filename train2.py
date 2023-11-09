import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import os

# Function to plot the ROC curve


def plot_roc(y_test, y_score, y_train, y_score_train, savepath):
    try:
        os.mkdir(savepath)
    except:
        pass
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr_train = dict()
    tpr_train = dict()
    roc_auc_train = dict()

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute micro-average ROC curve and ROC area
    fpr_train["micro"], tpr_train["micro"], _ = metrics.roc_curve(
        y_train.ravel(), y_score_train.ravel())
    roc_auc_train["micro"] = metrics.auc(
        fpr_train["micro"], tpr_train["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
             lw=lw, label='Test ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot(fpr_train["micro"], tpr_train["micro"], color='green',
             lw=lw, label='Training ROC curve (area = %0.2f)' % roc_auc_train["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(savepath + "/ROC.png")
    plt.show()


def get_dataframe(csv_name):
    df = pd.read_csv(csv_name)
    # 填補缺失資料
    # 如果沒花任何錢""

    df.loc[((df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']) == 0)&
           (df['CryoSleep'].isna()), 'CryoSleep'] = True
    # We start by filling the missing values with the mode of the object columns
    df.loc[(df['CryoSleep'].isna()), 'CryoSleep'] = False
    char_variables = list(df.select_dtypes(include=['object']).columns)

    # We remove the columns that we do not want to fill since it doesnt make sense
    char_variables.remove('Name')
    char_variables.remove('Cabin')

    for i in char_variables:
        df[i] = df[i].fillna(df[i].mode()[0])

    # We fill the missing values with the mean of the numeric columns
    numeric_variables = list(df.select_dtypes(
        include=['int64', 'float64']).columns)

    for col in numeric_variables:
        df.hist(column=col)
    # We first fill the money values based on the cryosleep values
    df['RoomService'] = np.where(df['CryoSleep'] == True, 0, df['RoomService'])
    df['FoodCourt'] = np.where(df['CryoSleep'] == True, 0, df['FoodCourt'])
    df['ShoppingMall'] = np.where(
        df['CryoSleep'] == True, 0, df['ShoppingMall'])
    df['Spa'] = np.where(df['CryoSleep'] == True, 0, df['Spa'])
    df['VRDeck'] = np.where(df['CryoSleep'] == True, 0, df['VRDeck'])

    for i in numeric_variables:
        df[i] = df[i].fillna(df[i].median())

    # Based on the information we have, we can create new columns:

    # New column with the passenger group
    df['PassengerGroup'] = df['PassengerId'].str.slice(0, 4)

    # 將Cabin的三個屬性分開
    df['Deck'] = df['Cabin'].str.split('/').str[0]
    df['Room'] = df['Cabin'].str.split('/').str[1]
    df['Side'] = df['Cabin'].str.split('/').str[2]

    # New colums to indicate if the passenger has a family member or not
    df['HasFamily'] = df['PassengerGroup'].isin(df['PassengerGroup'].value_counts()[
                                                df['PassengerGroup'].value_counts() > 1].index).astype('bool')

    # We create new variables with a boolean if passenger has spent money or not
    df['BoolRoom'] = df['RoomService'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    df['BoolFood'] = df['FoodCourt'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    df['BoolMall'] = df['ShoppingMall'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    df['BoolSpa'] = df['Spa'].apply(lambda x: 1 if x > 0 else 0).astype('bool')
    df['BoolVRDeck'] = df['VRDeck'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')

    # We create new variables with total money spent and average money spent
    money_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalBill'] = df[money_cols].sum(axis=1)
    # df['AverageBill'] = df[money_cols].mean(axis=1)

    # We create new variables with a boolean if passenger has spent money or not
    df['IsBill'] = df['TotalBill'] > 0
    df['CountBill'] = df[money_cols].replace(
        0, np.nan, inplace=False).count(axis=1, numeric_only=True)
    return df


if __name__ == '__main__':
    df = get_dataframe('./data/train.csv')

    # We create two new data frames, one with the features and the other with the target
    Features = df.loc[:, df.columns.difference(
        ['Transported', 'Name', 'Cabin', 'PassengerGroup'])]
    char_features = list(Features.select_dtypes(include=['object']).columns)
    char_features.remove('PassengerId')
    print(char_features)
    y = df['Transported']

    # Generate binary values using get_dummies
    dum_df = pd.get_dummies(Features[char_features], columns=char_features, prefix=[
                            'Deck_', 'Destination_', 'HomePlanet_', 'Room_', 'Side_'])
    X = Features.join(dum_df)
    X = X.drop(char_features, axis=1)
    X = X.drop('PassengerId', axis=1)
    X.to_csv("x.csv")
    # We split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1234)

    # Parameters for the grid search
    #ett = ExtraTreesClassifier()
    rdf = RandomForestClassifier()
    # Champion model parameters :  {'learning_rate': 0.25555555555555554, 'loss': 'exponential', 'max_depth': 5, 'min_samples_leaf': 0.001, 'min_samples_split': 0.26410526315789473, 'n_estimators': 100}
    param_grid_1 = {
        # "loss": ["exponential"],
        # "criterion":['gini', 'entropy'],
        "learning_rate": [0.07777777777777778],# np.linspace(0.05, 0.1, 10),
        # "min_samples_split": np.linspace(0.001, 5, 20),
        # "min_samples_leaf": np.linspace(0.001, 5, 20),
        "max_depth": [5],
        "n_estimators": [200]
    }

    param_grid_2 = {
        "loss": ["exponential"],
        # "criterion":['gini', 'entropy'],
        "learning_rate": np.linspace(0.1, 1.5, 10),
        "min_samples_split": np.linspace(0.001, 5, 20),
        "min_samples_leaf": np.linspace(0.001, 5, 20),
        "max_depth": [5],
        "n_estimators": [200]
    }

    # param_grid = {
    #     'kernel': ['poly', 'rbf', 'sigmoid'],
    #     'degree': [3, 4, 5],
    #     'coef0': [1, 2],
    #     'C': np.linspace(1, 5, 5),
    #     'max_iter': [500]
    # }

    # We fit a Gradient Boosting model with the train data
    # Champion model parameters :  {'learning_rate': 0.07777777777777778, 'max_depth': 5, 'n_estimators': 200}
    grid_search_1 = XGBClassifier(learning_rate=0.07777777777777778, max_depth=5, n_estimators=200)
    # grid_search_1 = GridSearchCV(estimator=xgb, param_grid=param_grid_1,
    #                            cv=3, n_jobs=-1, verbose=1)
    grid_search_1.fit(X_train, y_train)
    # print('Champion model parameters : ', grid_search_1.best_params_)

    # Champion model parameters :  {'learning_rate': 0.1, 'loss': 'exponential', 。'max_depth': 5, 'min_samples_leaf': 0.001, 'min_samples_split': 0.26410526315789473, 'n_estimators': 200}
    # gbc = GradientBoostingClassifier()
    # grid_search_2 = GridSearchCV(estimator=gbc, param_grid=param_grid_2,
    #                            cv=3, n_jobs=-1, verbose=1)
    # grid_search_2.fit(X_train, y_train)                          
    # print('Champion model parameters : ', grid_search_2.best_params_)
    grid_search = grid_search_1

    # grid_search = VotingClassifier(estimators=[
    #     ('xgb', grid_search_1),
    #     ('gbc', grid_search_2),
    # ], voting='soft')

    # grid_search.fit(X_train, y_train)
    # We predict the test data to check the accuracy
    
    # print('Champion model parameters : ', grid_search_2.best_params_)
    print('Test Score : ', grid_search.score(X_test, y_test))
    print('Train Score : ', grid_search.score(X_train, y_train))

    # We plot the ROC curve
    # plot_roc(y_test, grid_search.predict(X_test),
    #          y_train, grid_search.predict(X_train), "roc_xgb_5")

    test = get_dataframe('./data/test.csv')

    Features_ = test.loc[:, test.columns.difference(
        ['Transported', 'Name', 'Cabin', 'PassengerGroup'])]
    char_features = list(Features_.select_dtypes(include=['object']).columns)
    char_features.remove('PassengerId')
    print(char_features)

    # Generate binary values using get_dummies
    dum_test = pd.get_dummies(Features_[char_features], columns=char_features, prefix=[
                              'Deck_', 'Destination_', 'HomePlanet_', 'Room_', 'Side_'])
    X_ = Features_.join(dum_test)
    X_ = X_.drop(char_features, axis=1)
    X_ = X_.drop('PassengerId', axis=1)
    X_ = X_[X.columns]

    # Generate the predictions
    # print(grid_search.predict(X_))
    test['Transported'] = grid_search.predict(X)
    # Check predictions frame
    test['Transported'] = test['Transported'].astype('bool')
    # test[['PassengerId', 'Transported']]
    test[['PassengerId', 'Transported']].to_csv(
        'submission_rdf.csv', index=False)
