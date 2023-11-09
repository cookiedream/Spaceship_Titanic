from lib import *

# Filename
FILE_PREFIX = "./data/"
traindata = FILE_PREFIX + "train.csv"
testdata = FILE_PREFIX + "test.csv"

# 讀取訓練資料
train = pd.read_csv(traindata)
# train = pd.read_csv(test)
test = pd.read_csv(testdata)
dirpath = './images'
# 計算缺失值
# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Missing number', 'Missing rate'])
# print(missing_data)



# print(train.isnull().sum())
# train['CryoSleep']=train['CryoSleep'].fillna(False)
# train['VIP']=train['VIP'].fillna(False)
# train['Age']=train['Age'].fillna(train['Age'].mean())
# train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
# train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
# train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
# train['Spa']=train['Spa'].fillna(train['Spa'].mean())
# train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())
# train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
# train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
# train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
# train['Spa']=train['Spa'].fillna(train['Spa'].mean())
# train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())
# train['Destination']=train['Destination'].fillna('TRAPPIST-1e')
# train['HomePlanet']=train['HomePlanet'].fillna('Earth')
# train["Cabin"] = train["Cabin"].fillna(value="UNKNOWN") #還不知道可以補什麼 就給他 不知道吧
# Not sure why these features have NaN's, but we can (safely?) replace them with 0's.

# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Missing number', 'Missing rate'])
# print(train.isnull().sum())
# print(missing_data)

# 先把name的缺失值改成UNKNOWN
#====================================================================================
# train.Name = train.Name.fillna('UNKNOWN')
# train["CryoSleep"] = train["CryoSleep"].fillna(value=False)
# train["VIP"] = train["VIP"].fillna(value=False)
# train["VRDeck"] = train["VRDeck"].fillna(value=0)
# train["ShoppingMall"] = train["ShoppingMall"].fillna(value=0)
# train["RoomService"] = train["RoomService"].fillna(value=0)
# train["FoodCourt"] = train["FoodCourt"].fillna(value=0)
# train["Spa"] = train["Spa"].fillna(value=0)
# # weird ones
# train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)
# train["HomePlanet"] = train["HomePlanet"].fillna(value="UNKNOWN")
# train["Destination"] = train["Destination"].fillna(value="UNKNOWN")
# train["Age"] = train["Age"].fillna(value=0)
#====================================================================================
# #将训练集中缺失的Name填充为Unknown   
# train.Name = train.Name.fillna('Unknown')
# # print(train.isnull().sum())
 
# #将训练集中缺失的CryoSlee填充为False   
# train['CryoSleep']=train['CryoSleep'].fillna(False)
# # print(train.isnull().sum())
 
# #将训练集中缺失的VIP填充为False   
# train['VIP']=train['VIP'].fillna(False)
# # print(train.isnull().sum())
 
# #将训练集中缺失的Age填充为Age平均数   
# train['Age']=train['Age'].fillna(train['Age'].mean())
# # print(train.isnull().sum())
 
# #如果CryoSleep状态为True，那么将RoomService,FoodCourt,ShoppingMall,Spa,VRDeck填写为0  
# Expenses_columns = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
# train.loc[:,Expenses_columns]=train.apply(lambda x: 0 if x.CryoSleep == True else x,axis =1)
# # print(train.isnull().sum())
 
# #将训练集中缺失的RoomService填充为RoomService平均数   
# train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
 
# #将训练集中缺失的FoodCourt填充为FoodCourt平均数   
# train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
 
# #将训练集中缺失的ShoppingMall填充为ShoppingMall平均数   
# train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
 
# #将训练集中缺失的Spa填充为Spa平均数   
# train['Spa']=train['Spa'].fillna(train['Spa'].mean())
 
# #将训练集中缺失的VRDeck填充为VRDeck平均数   
# train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())

# #统计到达各个目的地的旅客的出发地的总数   
# analys = train.loc[:,['HomePlanet','Destination']]
# analys['numeric'] =1
# analys.groupby(['Destination','HomePlanet']).count()

# #将训练集中缺失的Destination填充为TRAPPIST-1e   
# #将训练集中缺失的HomePlanet填充为Earth平均数   
# train['Destination']=train['Destination'].fillna('TRAPPIST-1e')
# train['HomePlanet']=train['HomePlanet'].fillna('Earth')
# train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)
#====================================================================================
def get_dataframe(csv_name):
    train = pd.read_csv(csv_name)
    # 填補缺失資料
    # 如果沒花任何錢""

    train.loc[((train['RoomService'] + train['FoodCourt'] + train['ShoppingMall'] + train['Spa'] + train['VRDeck']) == 0)&
           (train['CryoSleep'].isna()), 'CryoSleep'] = True
    # We start by filling the missing values with the mode of the object columns
    train.loc[(train['CryoSleep'].isna()), 'CryoSleep'] = False
    char_variables = list(train.select_dtypes(include=['object']).columns)

    char_variables.remove('Name')
    char_variables.remove('Cabin')

    for i in char_variables:
        train[i] = train[i].fillna(train[i].mode()[0])

    numeric_variables = list(train.select_dtypes(
        include=['int64', 'float64']).columns)

    for col in numeric_variables:
        train.hist(column=col)
    # We first fill the money values based on the cryosleep values
    train['RoomService'] = np.where(train['CryoSleep'] == True, 0, train['RoomService'])
    train['FoodCourt'] = np.where(train['CryoSleep'] == True, 0, train['FoodCourt'])
    train['ShoppingMall'] = np.where(
        train['CryoSleep'] == True, 0, train['ShoppingMall'])
    train['Spa'] = np.where(train['CryoSleep'] == True, 0, train['Spa'])
    train['VRDeck'] = np.where(train['CryoSleep'] == True, 0, train['VRDeck'])


    for i in numeric_variables:
        train[i] = train[i].fillna(train[i].median())

    # Based on the information we have, we can create new columns:

    # New column with the passenger group
    train['PassengerGroup'] = train['PassengerId'].str.slice(0, 4)

    # 將Cabin的三個屬性分開
    train['Deck'] = train['Cabin'].str.split('/').str[0]
    train['Room'] = train['Cabin'].str.split('/').str[1]
    train['Side'] = train['Cabin'].str.split('/').str[2]

    # New colums to indicate if the passenger has a family member or not
    train['HasFamily'] = train['PassengerGroup'].isin(train['PassengerGroup'].value_counts()[
                                                train['PassengerGroup'].value_counts() > 1].index).astype('bool')

    train['BoolRoom'] = train['RoomService'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    train['BoolFood'] = train['FoodCourt'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    train['BoolMall'] = train['ShoppingMall'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    train['BoolSpa'] = train['Spa'].apply(lambda x: 1 if x > 0 else 0).astype('bool')
    train['BoolVRDeck'] = train['VRDeck'].apply(
        lambda x: 1 if x > 0 else 0).astype('bool')
    
    money_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    train['TotalBill'] = train[money_cols].sum(axis=1)
    # df['AverageBill'] = df[money_cols].mean(axis=1)

    # We create new variables with a boolean if passenger has spent money or not
    train['IsBill'] = train['TotalBill'] > 0
    train['CountBill'] = train[money_cols].replace(
        0, np.nan, inplace=False).count(axis=1, numeric_only=True)
    
    return train

#=====================================================================================================================

# def get_dataframe(train):
    
    
#     train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    
#     train.isnull().sum().sort_values(ascending=False)
    
#     label = "Transported"
#     train[label] = train[label].astype(int)
#     train['VIP'] = train['VIP'].astype(int)
#     train['CryoSleep'] = train['CryoSleep'].astype(int)
#     train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)
#     try:
#         train = train.drop('Cabin', axis=1)
#     except KeyError:
#         print("Field does not exist")
    
    
#     return train


# train = get_dataframe(train)
# # 計算缺失值
# total = train.isnull().sum().sort_values(ascending=False)
# percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
# missing_data = pd.concat([total, percent], axis=1, keys=['Missing number', 'Missing rate'])
# print(train.isnull().sum())
# print(missing_data)


if __name__ == '__main__':
    
    data = get_dataframe(traindata)

    # We create two new data frames, one with the features and the other with the target
    Features = data.loc[:, data.columns.difference(
        ['Transported', 'Name', 'Cabin', 'PassengerGroup'])]
    char_features = list(Features.select_dtypes(include=['object']).columns)
    char_features.remove('PassengerId')
    print(char_features)
    y = data['Transported']

    # Generate binary values using get_dummies
    dum_data = pd.get_dummies(Features[char_features], columns=char_features, prefix=[
                            'Deck_', 'Destination_', 'HomePlanet_', 'Room_', 'Side_'])
    X = Features.join(dum_data)
    X = X.drop(char_features, axis=1)
    X = X.drop('PassengerId', axis=1)
    X.to_csv("x.csv")
    # We split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1234)


#====================================================================================

train = get_dataframe(traindata)

try:
    os.mkdir(dirpath)
except:
    pass


plt.figure(figsize=(10, 5))
sns.histplot(data=train, x='Age', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)')
plt.savefig(dirpath + './Age_distribution.png')


fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(18, 12)
fig.subplots_adjust(wspace=0.3, hspace=0.3)
temp = train.fillna(-1)
sns.barplot(x = "HomePlanet", y= "Transported", data=temp, ax = ax[0][0])
sns.barplot(x = "CryoSleep", y= "Transported", data=temp, ax = ax[0][1])
sns.barplot(x = "VIP", y= "Transported", data=temp, ax = ax[1][0])
sns.barplot(x = "Destination", y= "Transported", data=temp, ax = ax[1][1])
plt.savefig(dirpath + './barplot.png')




# 過濾包含字串值的欄位並移除
numeric_columns = train.select_dtypes(include=[np.number])

# 處理包含字串值的欄位
for column in train.columns:
    if train[column].dtype == object:
        train = train.drop(column, axis=1)  # 移除包含字串值的欄位

# 計算相關性矩陣
corrmat = train.corr()

# 繪製相關性矩陣的熱圖
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True)
plt.savefig(dirpath + '/heatmap.png')




# train model 



corrmat = train.corr()
k = 6 
high_corr_values = corrmat.nlargest(k, 'Transported')['Transported'].index
high_corr_values = high_corr_values.drop('Transported')
# print(high_corr_values)

X = train[high_corr_values]
y = train['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
# 然后在测试数据上应用相同的变换
X_test = imputer.transform(X_test)

# 找到包含NaN值的行索引
nan_indices = np.isnan(X_train).any(axis=1)

# 删除包含NaN值的行
X_train = X_train[~nan_indices]
y_train = y_train[~nan_indices]


# model parameter
# sigmoid = 'learning_rate=0.07777777777777778, max_depth=5, n_estimators=200'

print("===================================================================================================")
# xgboost model
xgbc = XGBClassifier(learning_rate=0.07777777777777778, max_depth=5, n_estimators=200)
xgbc.fit(X_train, y_train)
xgbc_pred = xgbc.predict(X_test)
print("xgboost accuracy: {:.3f}".format(metrics.accuracy_score(y_test, xgbc_pred)))
# print("xgboost F1_score: {:.3f}".format(metrics.f1_score(y_test, xgbc_pred)))
# print("xgboost precision: {:.3f}".format(metrics.precision_score(y_test, xgbc_pred)))
# print("xgboost recall: {:.3f}".format(metrics.recall_score(y_test, xgbc_pred)))
print("===================================================================================================")
# Random forest model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print("Random forest accuracy: {:.3f}".format(metrics.accuracy_score(y_test, rfc_pred)))
# print("Random forest F1_score: {:.3f}".format(metrics.f1_score(y_test, rfc_pred)))
# print("Random forest precision: {:.3f}".format(metrics.precision_score(y_test, rfc_pred)))
# print("Random forest recall: {:.3f}".format(metrics.recall_score(y_test, rfc_pred)))
print("===================================================================================================")
# KNN model
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print("KNN accuracy: {:.3f}".format(metrics.accuracy_score(y_test, knn_pred)))
# print("KNN F1_score: {:.3f}".format(metrics.f1_score(y_test, knn_pred)))
# print("KNN precision: {:.3f}".format(metrics.precision_score(y_test, knn_pred)))
# print("KNN recall: {:.3f}".format(metrics.recall_score(y_test, knn_pred)))
print("===================================================================================================")
# Hist gradient boosting model
hgbc = HistGradientBoostingClassifier(learning_rate=0.07777777777777778, max_iter=75)
hgbc.fit(X_train,y_train)
hgbc_pred = hgbc.predict(X_test)
print("hgbc accuracy: {:.3f}".format(metrics.accuracy_score(y_test, hgbc_pred)))
# print("hgbc F1_score: {:.3f}".format(metrics.f1_score(y_test, hgbc_pred)))
# print("hgbc precision: {:.3f}".format(metrics.precision_score(y_test, hgbc_pred)))
# print("hgbc recall: {:.3f}".format(metrics.recall_score(y_test, hgbc_pred)))
print("===================================================================================================")
# SGD model
sgd = SGDClassifier()
sgd.fit(X_train,y_train)
sgd_pred = sgd.predict(X_test)
print("SGD accuracy: {:.3f}".format(metrics.accuracy_score(y_test, sgd_pred)))
# print("SGD F1_score: {:.3f}".format(metrics.f1_score(y_test, sgd_pred)))
# print("SGD precision: {:.3f}".format(metrics.precision_score(y_test, sgd_pred)))
# print("SGD recall: {:.3f}".format(metrics.recall_score(y_test, sgd_pred)))
print("===================================================================================================")
# SGD model
svm = SVC()
svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)
print("svm accuracy: {:.3f}".format(metrics.accuracy_score(y_test, svm_pred)))
# print("svm F1_score: {:.3f}".format(metrics.f1_score(y_test, svm_pred)))
# print("svm precision: {:.3f}".format(metrics.precision_score(y_test, svm_pred)))
# print("svm recall: {:.3f}".format(metrics.recall_score(y_test, svm_pred)))
print("===================================================================================================")
#使用GaussianNB模型进行预测并输出精度   
gnb = GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred = gnb.predict(X_test)
print("Gaussian NB accuracy: {:.3f}".format(metrics.accuracy_score(y_test,gnb_pred)))
print("===================================================================================================")
#使用MultinomialNB模型进行预测并输出精度   
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
mnb_pred = mnb.predict(X_test)
print("Multinomial NB accuracy: {:.3f}".format(metrics.accuracy_score(y_test,mnb_pred)))
print("===================================================================================================")
#使用BaggingClassifier模型进行预测并输出精度   
bag = BaggingClassifier()
bag.fit(X_train,y_train)
bag_pred = bag.predict(X_test)
print("Bagging accuracy: {:.3f}".format(metrics.accuracy_score(y_test,bag_pred)))
print("===================================================================================================")
#使用AdaBoostClassifier模型进行预测并输出精度   
abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
abc_pred = abc.predict(X_test)
print("Adaboost accuracy: {:.3f}".format(metrics.accuracy_score(y_test,abc_pred)))
print("===================================================================================================")
tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
tree_pred = tree.predict(X_test)
print("DecisionTreeClassifier accuracy: {:.3f}".format(metrics.accuracy_score(y_test,tree_pred)))