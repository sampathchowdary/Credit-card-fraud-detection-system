import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


print "Started"

df=pd.read_csv("creditcard.csv")
attr_list=['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
#for v in attr_list:
#    df[v].fillna(df[v].mean(), inplace=True)
x = df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']]
y = df['Class']
#Class Distribution
fig, ax = plt.subplots(1, 1)
ax.pie(df.Class.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'], colors=['y','r'])
plt.axis('equal')
plt.ylabel('')
plt.show()

#V1-V28
plt.figure(figsize=(6,28*4))
for i, col in enumerate(df[df.iloc[:,0:29].columns]):
    sns.distplot(df[col][df.Class == 1], bins=50, color='r')
    sns.distplot(df[col][df.Class == 0], bins=50, color='g')
    plt.xlabel('')
    plt.title('feature: ' + str(col))
    plt.show()


print "Completed"

#drop_list = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
