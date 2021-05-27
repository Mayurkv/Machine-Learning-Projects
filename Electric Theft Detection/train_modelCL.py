def main():
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE as SM
    from sklearn.cluster import KMeans
    from xgboost.sklearn import XGBClassifier
    from sklearn import  metrics
    
    basepath=r"D:\\Codedata\\"
    
    
    df=pd.read_csv(basepath + 'data.csv',delim_whitespace=False)
    
#    df['FLAG']=df['FLAG'].replace(to_replace = 1, value ='Theft') 
#    df['FLAG']=df['FLAG'].replace(to_replace = 0, value ='Normal') 
    ##############################################################################
    l=df.columns
    la=['CONS_NO','FLAG']
    lb=[]
    for i in l:
        if i not in la:
            lb.append(i)
            
            
    import datetime
    dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
    #dates.sort()
    fdates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]
    
    fdates.insert(0,"FLAG")
    fdates.insert(0,"CONS_NO")
    df.columns=fdates
    
    
    
    import datetime
    dates = [datetime.datetime.strptime(ts, "%Y/%m/%d") for ts in lb]
    dates.sort()
    sorteddates = [datetime.datetime.strftime(ts, "%Y/%m/%d") for ts in dates]
    
    
    
    cols=df.columns.tolist()[0:2]+sorteddates
    df=df[cols]
    
    
    ######Process NAN value======================================================
    
    ####For first two col
    df1=df
    
    l=df["2014/01/01"]
    l1=df["2014/01/02"]
    l=np.asarray(l).tolist()
    l1=np.asarray(l1).tolist()
    
    l2=[]
    for i in range(len(l)):
        if math.isnan(l[i]):
            if math.isnan(l1[i]):
                l2.append(0)
            else:
                l2.append(l1[i]/2)
        else:
            l2.append(l[i])
            
    df1["2014/01/01"]=l2
    
    #############################################################################
    ####for last two days
    
    print(df.columns[-1],df.columns[-2])
    
    l=df["2016/10/31"]
    l1=df["2016/10/30"]
    l=np.asarray(l).tolist()
    l1=np.asarray(l1).tolist()
    
    l2=[]
    for i in range(len(l)):
        if math.isnan(l[i]):
            if math.isnan(l1[i]):
                l2.append(0)
            else:
                l2.append(l1[i]/2)
        else:
            l2.append(l[i])
    df1["2016/10/31"]=l2
    
    #############################################################################
    #for rest of the columns
    
    l=df.columns
    la=['CONS_NO','FLAG']
    lb=[]
    for i in l:
        if i not in la:
            lb.append(i)
    
    l=df.columns
    la=['CONS_NO','FLAG']
    lbx=[]
    for i in l:
        if i not in la:
            lbx.append(i)
          
    
    for i in range(1,len(lb)-1):
        l=np.asarray(df[lb[i]]).tolist()
        l1=np.asarray(df[lb[i-1]]).tolist()
        l2=np.asarray(df[lb[i+1]]).tolist()
        l3=[]
        for j in range(len(l)):
            if math.isnan(l[j]):
                if math.isnan(l1[j])==False and math.isnan(l2[j])==False:
                    l3.append((l1[j]+l2[j])/2)
                else:
                    l3.append(0)
            else:
                l3.append(l[j])
        df1[lb[i]]=l3
        
    #############################################################################
    #Model  Training===========================================================
    
    ###############################################################################
    
    X=df1[lbx[500:]]
    Y=df1["FLAG"]
    
    
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
    ###########==================================================================
    
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 43)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

        
    clf=KMeans(n_clusters=10,random_state=43)
    clf.fit(X)
    
    centroids=clf.cluster_centers_
    labels=clf.labels_
    #colors=["g.","r.","c.","b.","k.","o."]
    
    t=clf.predict(X)
    
    t=t.tolist()
    
    unique_elements, counts_elements = np.unique(t, return_counts=True)
    print(unique_elements, counts_elements )
    
    #############################################################################
    
    c=0
    l=[]
    
    for i in range(len(t)):
        if Y[i]==1:
            c+=1
            l.append(t[i])
    
    
    #c=0
    #
    #for i in l:
    #    if i==0:
    #        c+=1
    
    df1["labels"]=labels
    
    X["labels"]=labels
    
    orig=df1[df1["labels"]==0]
    orig=orig.drop(["labels"],axis=1)
    
    
    X1=X[X["labels"]==0]
    
    X1=X1.drop(["labels"],axis=1)
    len(X1)
    
    
    #print(X1.iloc[5])
    l=[]
    for i in range(len(X1)): # range(41816):
        l.append(X1.iloc[i])
        
    
    t1=list((np.array(l) - np.array(centroids[0]))**2)
    
    l2=[]
    for i in t1:
        l2.append(np.sqrt(np.sum(i)))
        
    mean=np.mean(l2)
    
    Y1=Y[X["labels"]==0]
    
    Y1=Y1.tolist()
    
    c=0
    l3=[]
    for i in range(len(X1)):    #range(41809):
        if Y1[i]==1:
            l3.append(l2[i])
    
    #out of 3371  2716 with distance greater than 100.
    X1["distance"]=l2
    
    X1["theft"]=Y1
    
    orig["distance"]=l2
    orig["theft"]=Y1
    
    X2=X1
    len(l2)
    
    
    c=0
    for i in range(len(l2)):
        if l2[i]<80 and Y1[i]==0:
            c+=1
    
    c=0
    h=[]
    for i in range(len(l2)):
        f=0
        if (l2[i]<=70 and Y1[i]==0) or (l2[i]>100 and Y1[i]==1):
            f=1
        if f==0:
            h.append(int(i))
    #X2=X2.drop(h,axis=1)
            
    X3=X2.drop(X2.index[h])
    
    orig=orig.drop(orig.index[h])
    
    #Yf=X3["theft"]
    #X3=X3.drop("theft",axis=1)
    Ecd=X3["distance"]
    
    X3=X3.drop("distance",axis=1)
    orig=orig.drop("distance",axis=1)
    
    lb=X3.columns.tolist()
    l1=lb
    
    
    #Yf=Yf.tolist()
    Yf=X3["theft"]
    
    Yf=Yf.tolist()
    
    #c=0
    #for i in Yf:
    #    if i==0:
    #        c+=1
    #
    #len(Yf)
    
    
    plt.plot(l1,X3.iloc[3013][l1])
    
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(32,24), dpi=80, facecolor='w', edgecolor='k')
    axes = plt.gca()
    fig = plt.gcf()
    for i in range(4000,4005):
        plt.plot(l1,X3.iloc[i][l1],c='b')
        plt.plot(l1,X3.iloc[i-2000][l1],c='r')
    fig.savefig(basepath + 'ElectricityData1.png', dpi=200) 
    
    figure(num=None, figsize=(24,16), dpi=200, facecolor='w', edgecolor='k')
    axes = plt.gca()
    fig = plt.gcf()
    for i in range(3000,3010):
        plt.plot(l1,X3.iloc[i][l1],c='r')
        plt.plot(l1,X3.iloc[i+1000][l1],c='k')
    fig.savefig(basepath + 'ElectricityData2.png', dpi=200) 
        
        
        
    Ecd=Ecd.tolist()
    
    
    #Ecd[2008]
    #ecd less than 70 is 0 >100 is 1
    
    
    ## l3=[]
    l4=[]
    for i in range(len(l2)):
        if l2[i]<=70 and Y1[i]==0:
            l3.append(X1.iloc[i])
        elif l2[i]>=100 and Y1[i]==1:
            l4.append(X1.iloc[i])
            
    
    X_train,X_test,Y_train,Y_test=train_test_split(X3,Yf,test_size=0.20)
    
    
    sm=SM(random_state=42)
    #X_tr,Y_tr=sm.fit_sample(X_train,Y_train)
    X_tr,Y_tr=sm.fit_sample(X_train,Y_train)
    
    df1[df1["FLAG"]==1].describe()
    
    
    #############################################################################
    # fit model no training data
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'learning_rate': 1.0,
        'silent': True,  #1.0,
        'n_estimators': 5
    }
    model = XGBClassifier(**params).fit(X_train, Y_train)
    
    model.fit(X_train, Y_train)
    
    
    y_pred = model.predict(X_test)
#    predictions = [round(value) for value in y_pred]
    
    
    # how did our model perform?
    count_misclassified = (Y_test != y_pred).sum()
    A='Misclassified samples: {}'.format(count_misclassified)
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    B="Accuracy: %.2f%%" % (accuracy * 100.0)
    
    
    orig['CONS_NO'] =orig['CONS_NO'].str.strip()

    orig['FLAG']=orig['FLAG'].replace(to_replace = 1, value ='THEFT') 
    orig['FLAG']=orig['FLAG'].replace(to_replace = 0, value ='NORMAL') 
    
    orig['theft']=orig['theft'].replace(to_replace = 1, value ='THEFT') 
    orig['theft']=orig['theft'].replace(to_replace = 0, value ='NORMAL') 

    from sklearn.utils import shuffle
    orig = shuffle(orig)

    orig.to_csv(basepath+ "Result.csv",index=False)
    
    return B
    import csv
    
    
    
    
            
    #############################################################################
    #from sklearn.ensemble import RandomForestRegressor
    #
    #regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    #regressor.fit(X_train, Y_train)
    #y_pred = regressor.predict(X_test)
    #
    #predictions = [round(value) for value in y_pred]
    #
    #
    #accuracy = metrics.accuracy_score(Y_test, predictions)
    #print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #
    ####For Graph display############################
#main()