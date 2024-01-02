from lib import *

# Filename
FILE_PREFIX = "./data/"
dirpath = './images/'
# 讀取訓練資料
train_df = pd.read_csv( FILE_PREFIX + "train.csv" )

test_df = pd.read_csv( FILE_PREFIX + "test.csv" )


# PassengerId、HomePlanet、CryoSleep、Cabin、Destination、Age、VIP、RoomService、FoodCourt、ShoppingMall、Spa、VRDeck、Name、Transported

# print( train.columns.values )
# print ( train.dtypes)
# print(train.head(100))


# 計算缺失值
# train_missing = train_df.isnull().sum()
# test_missing = test_df.isnull().sum()

# print('training loss data:')
# print(train_missing)
# print()
# print('testing loss data:')
# print(test_missing)

# copy of train and test data

train_df_1 = train_df.copy()
test_df_1 = test_df.copy()



train_df_1[["CabinDeck", "CabinNo.", "CabinSide"]] = train_df_1["Cabin"].str.split('/', expand = True)

combine = [train_df_1, test_df_1]

# print(train.info())
# print('='*50)
# print(test.info())

# print(train_df.describe())
# print('='*150)
# print(train_df.describe(include=['O']))


for dataset in combine:    
    dataset['HomePlanet'].fillna(dataset['HomePlanet'].mode()[0],inplace = True)
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['CryoSleep'].fillna(dataset['CryoSleep'].mode()[0],inplace = True)
    dataset['Destination'].fillna(dataset['Destination'].mode()[0],inplace = True)
    dataset['VIP'].fillna(dataset['VIP'].mode()[0],inplace = True)
    dataset['RoomService'].fillna(dataset['RoomService'].median(), inplace = True)
    dataset['FoodCourt'].fillna(dataset['FoodCourt'].median(), inplace = True)
    dataset['ShoppingMall'].fillna(dataset['ShoppingMall'].median(), inplace = True)
    dataset['Spa'].fillna(dataset['Spa'].median(), inplace = True)
    dataset['VRDeck'].fillna(dataset['VRDeck'].median(), inplace = True)



train_missing = train_df_1.isnull().sum()
test_missing = test_df_1.isnull().sum()


# 補缺失值之後再次計算缺失值
# print('training loss data:')
# print(train_missing)
# print('='*150)
# print('testing loss data:')
# print(test_missing)


# Categorical features
cat_feats=['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

# Plot categorical features
fig=plt.figure(figsize=(10,15))
for i, var_name in enumerate(cat_feats):
    ax=fig.add_subplot(4,1,i+1)
    sns.countplot(data=train_df, x=var_name, axes=ax, hue='Transported')
    ax.set_title(var_name)
fig.tight_layout()  # Improves appearance a bit
plt.savefig( dirpath + 'Categorical_features.png' )



train_df_1 = pd.get_dummies(train_df_1, columns = ['HomePlanet','Destination','CryoSleep'])
test_df_1 = pd.get_dummies(test_df_1,columns = ['Destination'])

# train_df_1[["CabinDeck", "CabinNo.", "CabinSide"]] = train_df_1["Cabin"].str.split('/', expand = True)
# VIP = train_df_1[["VIP", "Transported"]].groupby(['VIP'], as_index=False).mean().sort_values(by='Transported', ascending=False)
# print(VIP)
# print(Transported)
# print(train_df_1)
# train_df_1 = pd.get_dummies(train_df_1, columns = ['VIP'])
# train_df_1['VIP'] = LabelEncoder().fit_transform(train_df_1['VIP'])

# 將 True 轉換為 1，False 轉換為 0
# train_df_1['VIP'] = train_df_1['VIP'].astype(int)
# test_df_1['VIP'] = test_df_1['VIP'].astype(int)

# 使用 get_dummies 進行 One-Hot Encoding
# train_df_1 = pd.get_dummies(train_df_1, columns=['VIP'], prefix='VIP')
# test_df_1 = pd.get_dummies(test_df_1, columns=['VIP'], prefix='VIP')
side_map = {'P':1,'S':0}
for dataset in combine:
    train_df_1['VIP'] = train_df_1['VIP'].astype(int)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    dataset['Cabin'].fillna('Z/9999/Z', inplace=True)
    
    
    dataset['deck'] = dataset['Cabin'].apply(lambda x:str(x)[:1])
    dataset['num'] = dataset['Cabin'].apply(lambda x:x.split('/')[1])
    dataset['num'] = dataset['num'].astype(int)
    dataset['side'] = dataset['Cabin'].apply(lambda x:str(x)[-1:])
    dataset['deck'].fillna(dataset['deck'].mode()[0],inplace=True)
    dataset['num'].fillna(dataset['num'].mode()[0],inplace=True)
    dataset['side'].fillna(dataset['side'].mode()[0],inplace=True)
    
    
    dataset['side'] = dataset['side'].map(side_map)
    dataset['side'].fillna(dataset['side'].mode()[0],inplace=True)
    
    # dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    # dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 31), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 47), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 47) & (dataset['Age'] <= 63), 'Age'] = 3
    # dataset.loc[(dataset['Age'] > 63), 'Age'] = 4
    # dataset['Age'] = dataset['Age'].astype(int)
    # print(dataset['VIP'].dtype)
    # print(dataset['VIP'].unique())

    

    
# train_df_1[['AgeBin','Transported']].groupby(['AgeBin'],as_index=False).mean().sort_values(by='Transported',ascending=False)

# train_df_1 = train_df_1.drop(['Cabin'],axis=1)
# test_df_1 = test_df_1.drop(['Cabin'],axis=1)

    
# train_df_1[['deck','Transported']].groupby(['deck'],as_index=False).mean().sort_values(by='Transported',ascending=False)
# for train_df_1 in combine:   
#     train_df_1['AgeBin'] = pd.cut(train_df_1['Age'].astype(int), 5)

print(train_df_1.columns.values)
# print(train_df_1.head())
# print(train_df_1.rpow)
# print('='*150)
# print(train_df.rpow)
# train_missing = train_df_1.isnull().sum()
# print(train_missing)




# # drop features created during EDA
# train_df_2 = train_df_1.copy()
# train_df_2 = train_df_2.drop(["PassengerGroup",
#                             "CabinDeck",
#                             "CabinNo.",
#                             "CabinSide",
#                             "FamilyName",
#                             "NoRelatives",
#                             "NoInPassengerGroup",
#                             "AgeCat",
#                             "FamilySizeCat", 
#                             "TotalSpendings"], axis = 1)

# # save target variable  in train train_df_1 and save it in target
# target = train_df_2["Transported"]

# # save test PassengerId in test_id
# test_id = test_df_1["PassengerId"]

# # drop PassengerId  variable from the train set
# train_df_3 = train_df_2.drop(["PassengerId"], axis = 1)

# # join the train and test set
# data = pd.concat([train_df_3, test], axis = 0).reset_index(drop = True)

# print(data.shape)

# 取得所有資料類別的欄位名稱
# category_columns = train_df_1[['HomePlanet', 'CryoSleep', 'Destination', 'VIP']].select_dtypes(include='object').columns

# 迴圈製作條形圖
# for column in category_columns:
#     plt.figure()
#     normalized_data = train_df_1[column].value_counts(normalize=True).reset_index()
#     normalized_data.columns = [column, f'Normalized {column} Count']
#     ax = sns.barplot(x=column, y=f'Normalized {column} Count', data=normalized_data, color='skyblue')
#     ax.set_xlabel(column)
#     ax.set_ylabel(f'Normalized {column} Count')
#     plt.savefig(dirpath + f'{column}_normalized.png')
    
    
    # 可視化獨立分類特徵
# train_df_1[["CabinDeck", "CabinNo.", "CabinSide"]] = train_df_1["Cabin"].str.split('/', expand = True)

# category_columns_1 = train_df_1[['Transported', 'CabinDeck', 'CabinNo.', 'CabinSide']].select_dtypes(include='object').columns

# for column in category_columns_1:
#     plt.figure()
#     ax = sns.countplot(x=column, data=train_df_1, color='b')
#     ax.set_xlabel(column)
#     ax.set_ylabel(f'{column} Count')
#     plt.savefig(dirpath + f'{column}.png')
    
    

# plt.figure(figsize=(16, 5))

# plt.subplot(121)
# sns.distplot(train_df_1['Age'])
# plt.title('Distribution of Age')

# plt.savefig( dirpath + 'Age.png' )
