from train import *
from lib import *
 
# #导出文件   
# test_ids = test["PassengerId"]
 
# from sklearn.preprocessing import LabelEncoder
# categorical_values_test = test.select_dtypes(include=['object']).columns
 
# for i in categorical_values_test:
# 	lbl = LabelEncoder()
# 	lbl.fit(list(test[i].values))
# 	test[i] = lbl.transform(list(test[i].values))
 
# #由于HistGradientBoostingClassifier模型预测精度最高，因此使用HistGradientBoostingClassifier所预测的文件   
# real_predictions = hgbc.predict(test[high_corr_values])
# print(len(test))
# # print(len(test.PassengerId))
 
# test["PassengerId"] = test_ids
 
# real_predictions = list(map(bool,real_predictions))
 
# output = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': real_predictions})
# output.to_csv(FILE_PREFIX + 'result.csv', index=False)
if __name__ == '__main__':

    # Parameters for the grid search
    #ett = ExtraTreesClassifier()
    # rdf = RandomForestClassifier()
    # Champion model parameters :  {'learning_rate': 0.25555555555555554, 'loss': 'exponential', 'max_depth': 5, 'min_samples_leaf': 0.001, 'min_samples_split': 0.26410526315789473, 'n_estimators': 100}
    # param_grid_1 = {
    #     # "loss": ["exponential"],
    #     # "criterion":['gini', 'entropy'],
    #     "learning_rate": [0.07777777777777778],# np.linspace(0.05, 0.1, 10),
    #     # "min_samples_split": np.linspace(0.001, 5, 20),
    #     # "min_samples_leaf": np.linspace(0.001, 5, 20),
    #     "max_depth": [5],
    #     "n_estimators": [200]
    # }

    # param_grid_2 = {
    #     "loss": ["exponential"],
    #     # "criterion":['gini', 'entropy'],
    #     "learning_rate": np.linspace(0.1, 1.5, 10),
    #     "min_samples_split": np.linspace(0.001, 5, 20),
    #     "min_samples_leaf": np.linspace(0.001, 5, 20),
    #     "max_depth": [5],
    #     "n_estimators": [200]
    # }

    # param_grid = {
    #     'kernel': ['poly', 'rbf', 'sigmoid'],
    #     'degree': [3, 4, 5],
    #     'coef0': [1, 2],
    #     'C': np.linspace(1, 5, 5),
    #     'max_iter': [500]
    # }

    # We fit a Gradient Boosting model with the train data
    # Champion model parameters :  {'learning_rate': 0.07777777777777778, 'max_depth': 5, 'n_estimators': 200}
    # grid_search_1 = XGBClassifier(learning_rate=0.07777777777777778, max_depth=5, n_estimators=200)
    # grid_search_1 = GridSearchCV(estimator=xgb, param_grid=param_grid_1,
    #                            cv=3, n_jobs=-1, verbose=1)
    # grid_search_1.fit(X_train, y_train)
    # print('Champion model parameters : ', grid_search_1.best_params_)

    # Champion model parameters :  {'learning_rate': 0.1, 'loss': 'exponential', 。'max_depth': 5, 'min_samples_leaf': 0.001, 'min_samples_split': 0.26410526315789473, 'n_estimators': 200}
    # gbc = GradientBoostingClassifier()
    # grid_search_2 = GridSearchCV(estimator=gbc, param_grid=param_grid_2,
    #                            cv=3, n_jobs=-1, verbose=1)
    # grid_search_2.fit(X_train, y_train)                          
    # print('Champion model parameters : ', grid_search_2.best_params_)
    # grid_search = grid_search_1

    # grid_search = VotingClassifier(estimators=[
    #     ('xgb', grid_search_1),
    #     ('gbc', grid_search_2),
    # ], voting='soft')

    # grid_search.fit(X_train, y_train)
    # We predict the test data to check the accuracy
    
    # print('Champion model parameters : ', grid_search_2.best_params_)
    # print('Test Score : ', grid_search.score(X_test, y_test))
    # print('Train Score : ', grid_search.score(X_train, y_train))

    # We plot the ROC curve
    # plot_roc(y_test, grid_search.predict(X_test),
    #          y_train, grid_search.predict(X_train), "roc_xgb_5")
    
    
    # Generate binary values using get_dummies
    
    # We create two new data frames, one with the features and the other with the target
    Features = train.loc[:, train.columns.difference(
        ['Transported', 'Name', 'Cabin', 'PassengerGroup'])]
    char_features = list(Features.select_dtypes(include=['object']).columns)
    char_features.remove('PassengerId')
    print(char_features)
    y = train['Transported']
    
    
    dum_df = pd.get_dummies(Features[char_features], columns=char_features, prefix=[
                            'Deck_', 'Destination_', 'HomePlanet_', 'Room_', 'Side_'])
    X = Features.join(dum_df)
    X = X.drop(char_features, axis=1)
    X = X.drop('PassengerId', axis=1)
    X.to_csv(FILE_PREFIX + "./test1.csv")
    
    

    test = get_dataframe(testdata)

    Features_ = test.loc[:, test.columns.difference(
        ['Transported', 'Name', 'Cabin', 'PassengerGroup'])]
    char_features = list(Features_.select_dtypes(include=['object']).columns)
    
if 'PassengerId' in char_features:
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
    test['Transported'] = hgbc.predict(X)
    # Check predictions frame
    test['Transported'] = test['Transported'].astype('bool')
    # test[['PassengerId', 'Transported']]
    test[['PassengerId', 'Transported']].to_csv(
        'submission_rdf.csv', index=False)