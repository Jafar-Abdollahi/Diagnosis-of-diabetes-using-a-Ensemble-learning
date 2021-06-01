from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import tkinter
from tkinter import messagebox


def predict():
        window = Tk()
        window.withdraw()
        window.geometry('275x260+500+300')
        window.title("Result=")
        ##############################
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import sklearn
        from sklearn.model_selection import train_test_split
        import numpy as np
        import pandas as pd
        import re
        import sklearn
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import VotingClassifier
        from sklearn import ensemble
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn import tree
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn import svm
        from sklearn.ensemble import ExtraTreesClassifier    
        from sklearn import model_selection
        from sklearn.ensemble import VotingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn import model_selection
        #%matplotlib inline
        
        Diabetic=pd.read_csv('Diabetic.csv')
        X=Diabetic[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
        y=Diabetic['Outcome']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=0)

        ###################
        #Load Algorithms Mchine Learning
        from sklearn.ensemble import RandomForestClassifier
        rfc=RandomForestClassifier(n_estimators=12)
        rfc.fit(X,y)

        from sklearn.neural_network import MLPClassifier
        MLP=MLPClassifier(alpha=0.01,hidden_layer_sizes=(100,),validation_fraction=0.9,shuffle=False)
        MLP.fit(X,y)

        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier(n_neighbors=2,p=1,leaf_size=738)
        knn.fit(X_train,y_train)
        
        from sklearn.ensemble import AdaBoostClassifier
        ABC=AdaBoostClassifier(n_estimators=67)
        ABC.fit(X,y)

        from sklearn.tree import DecisionTreeClassifier
        DTree=tree.DecisionTreeClassifier(criterion='gini',random_state=60)
        DTree.fit(X,y)

        from sklearn.naive_bayes import GaussianNB
        GNB=GaussianNB()
        GNB.fit(X_train,y_train)

        from sklearn.ensemble import GradientBoostingClassifier
        gbc=GradientBoostingClassifier(n_estimators=4,max_depth=11,random_state=2,min_samples_leaf=2)
        gbc.fit(X_train,y_train)

        from sklearn import svm
        SVM=svm.SVC(kernel='rbf', probability=True)
        SVM.fit(X_train,y_train)

        from sklearn.ensemble import ExtraTreesClassifier
        ETC=ExtraTreesClassifier(random_state = 0, n_jobs = -1, n_estimators = 100, max_depth = 3)
        ETC.fit(X_train,y_train)
        ##############################
        #Predict Diyabetic
        Diabetic_prediction_rfc=rfc.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]]) 
        lookup_Diabetic_name_rfc=[Diabetic_prediction_rfc[0]]
        print('[1]','RFC=',lookup_Diabetic_name_rfc)
        
        #Diabetic_prediction_MLP=MLP.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        #lookup_Diabetic_name_MLP=[Diabetic_prediction_MLP[0]]
        #print('[2]','MLP=',lookup_Diabetic_name_MLP)
        
        Diabetic_prediction_knn=knn.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        lookup_Diabetic_name_knn=[Diabetic_prediction_knn[0]]
        print('[3]','knn=',lookup_Diabetic_name_knn)
        
        
        Diabetic_prediction_ABC=ABC.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        lookup_Diabetic_name_ABC=[Diabetic_prediction_ABC[0]]
        print('[4]','AdaBoost=',lookup_Diabetic_name_ABC)
        
        Diabetic_prediction_TREE=DTree.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        lookup_Diabetic_name_TREE=[Diabetic_prediction_TREE[0]]
        print('[5]','Tree=',lookup_Diabetic_name_TREE)
        
        #Diabetic_prediction_GNB=GNB.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        #lookup_Diabetic_name_GNB=[Diabetic_prediction_GNB[0]]
        #print('[6]','GaussianNB=',lookup_Diabetic_name_GNB)
        
        Diabetic_prediction_GradientBoosting=gbc.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        lookup_Diabetic_name_GBC=[Diabetic_prediction_GradientBoosting[0]]
        print('[7]','GradientBoosting=',lookup_Diabetic_name_GBC)
        
        
        Diabetic_prediction_SVM=SVM.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        lookup_Diabetic_name_SVM=[Diabetic_prediction_SVM[0]]
        print('[8]','SVM=',lookup_Diabetic_name_SVM)
        
        
        Diabetic_prediction_ETC=ETC.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]]) 
        lookup_Diabetic_name_ETC=[Diabetic_prediction_ETC[0]]
        print('[9]','ETC=',lookup_Diabetic_name_ETC)
        ##############################
        print(window,'-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        predicted=rfc.predict(X_test)
        accurace_rfc=np.mean(predicted==y_test)
        print('[1]','accuracy for RandomForestClassifier =%.2f'%accurace_rfc)
        
        predicted=knn.predict(X_test)
        accurace_knn=np.mean(predicted==y_test)
        print('[2]','accuracy for KNeighborsClassifier =%.2f'%accurace_knn)
        
        
        predicted=MLP.predict(X_test)
        accurace_MLP=np.mean(predicted==y_test)
        print('[3]','accuracy for MLPClassifier =%.2f'%accurace_MLP)
        
        predicted=ABC.predict(X_test)
        accurace_ABC=np.mean(predicted==y_test)
        print('[4]','accuracy for AdaBoostClassifier =%.2f'%accurace_ABC)
        
        
        predicted=DTree.predict(X_test)
        accurace_Dtree=np.mean(predicted==y_test)
        print('[5]','accuracy for tree =%.2f'%accurace_Dtree)
        
        
        predicted=GNB.predict(X_test)
        accurace_GNB=np.mean(predicted==y_test)
        print('[6]','accuracy for GaussianNB =%.2f'%accurace_GNB)
        
        
        predicted=gbc.predict(X_test)
        accurace_gbc=np.mean(predicted==y_test)
        print('[7]','accuracy for GradientBoostingClassifier =%.2f'%accurace_gbc)
        
        predicted=SVM.predict(X_test)
        accurace_SVM=np.mean(predicted==y_test)
        print('[8]','accuracy for Support Vector Machine =%.2f'%accurace_SVM)
        
        predicted=ETC.predict(X_test)
        accurace_ETC=np.mean(predicted==y_test)
        print('[9]','accuracy for ExtraTreesClassifier =%.2f'%accurace_ETC)

        
        #Accuracy
        print(window,'-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-')
        ###########################
        #Staking_Classifier
        import time
        from sklearn import model_selection
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB 
        from sklearn.ensemble import RandomForestClassifier
        from mlxtend.classifier import StackingClassifier
        import numpy as np

        clf1 = RandomForestClassifier(n_estimators=12)
        clf2 = KNeighborsClassifier(n_neighbors=2,p=1,leaf_size=738)
        clf3 = MLPClassifier(alpha=0.01,hidden_layer_sizes=(100,),validation_fraction=0.9,shuffle=False)
        clf4 = AdaBoostClassifier(n_estimators=67)
        clf5 = tree.DecisionTreeClassifier(criterion='gini')
        clf6 = GaussianNB()
        clf7 = GradientBoostingClassifier(n_estimators=4,max_depth=11,min_samples_leaf=2)
        clf8 = svm.SVC()
        clf9 = ExtraTreesClassifier(random_state = 0, n_jobs = -1, n_estimators = 100, max_depth = 3)

        
        print()
        print()
        print()
        print()
        print()
        print()

        
        print("RESULT.............................RESULT")
        import time
        import stacked_generalization
        start=time.time()
        from stacked_generalization import StackedGeneralizer
        import stacked_generalization
        from mlxtend.classifier import StackingClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
        from sklearn import datasets, metrics

        # Stage 1 model
        bclf = LogisticRegression()

        # Stage 0 models
        clfs = [clf1,clf2,clf3,clf4,clf4,clf6,clf7,clf8,clf9]


        # same interface as scikit-learn
        sl = StackingClassifier(classifiers=clfs,meta_classifier=bclf)
        sl.fit(X,y)
        score = metrics.accuracy_score(y, sl.predict(X))
        print("Accuracy: %f" % score)
        print("RESULT.............................RESULT")


        #################################### Print IN Form
        result1 =ttk.Label(window, text='Result Accuracy').pack()
        lab=Label(window,text=score).place(x=200,y=0)
        messagebox.showerror("Result Accuracy",score)
        ##########################################
        #result2 =ttk.Label(window, text='Diagnosis Stack=').pack()
        #Diabetic_prediction_SL=sl.predict([[first_name.get(),last_name.get(),Family.get(),Contact.get(),Address.get(),NFater.get(),NMader.get(),House.get()]])
        #lookup_Diabetic_name_Stack=[Diabetic_prediction_SL[0]]
        #lab=Label(window,text=lookup_Diabetic_name_Stack).place(x=210,y=17)
        #print('[Result]','Stack=',lookup_Diabetic_name_Stack)
        messagebox.showwarning("Diagnosis Stack=" )#,lookup_Diabetic_name_Stack)
        ###############################################
        
        result4 =ttk.Label(window, text='predict Result All=').pack()
        a=lookup_Diabetic_name_rfc
        #b=lookup_Diabetic_name_MLP
        c=lookup_Diabetic_name_knn
        d=lookup_Diabetic_name_ABC
        e=lookup_Diabetic_name_TREE
        #f=lookup_Diabetic_name_GNB
        g=lookup_Diabetic_name_GBC
        h=lookup_Diabetic_name_SVM
        i=lookup_Diabetic_name_ETC
        #j=lookup_Diabetic_name_Stack
        resultD=(a+c+d+e+g+h+i)
        #resultD=(a+b+c+d+e+f+g+h+i+j)
        print('resultD=',resultD)
        import numpy as np
        import numpy
        import collections
        prediict=collections.Counter(resultD)
        lab=Label(window,text=prediict).place(x=210,y=38)
        result4 =ttk.Label(window, text='Nagative=').pack()
        Nagative=collections.Counter(prediict)[0]
        result4 =ttk.Label(window, text='Positive=').pack()
        Positive=collections.Counter(prediict)[1]

        lab=Label(window,text=Nagative).place(x=210,y=58)
        lab=Label(window,text=Positive).place(x=210,y=75)
        
        print('Nagative',collections.Counter(prediict)[0])
        print('Positive',collections.Counter(prediict)[1])
        print(prediict)
        messagebox.showinfo("predict Result All=",prediict)
        window.mainloop()
        ########################################### Print IN Form

       
root = Tk()
root.geometry('570x300+500+300')
root.title("program")


ttk.Label(root, text="Pregnancies:").grid(row=0, column=0)
first_name = ttk.Entry(root)
first_name.grid(row=0, column=1)

ttk.Label(root, text="Glucose :").grid(row=1, column=0)
last_name = ttk.Entry(root)
last_name.grid(row=1, column=1)

ttk.Label(root, text="BloodPressure :").grid(row=2, column=0)
Family = ttk.Entry(root)
Family.grid(row=2, column=1)


ttk.Label(root, text="SkineThickness :").grid(row=3, column=0)
Contact = ttk.Entry(root)
Contact.grid(row=3, column=1)


ttk.Label(root, text="Insulin :").grid(row=4, column=0)
Address = ttk.Entry(root)
Address.grid(row=4, column=1)


ttk.Label(root, text="BMI :").grid(row=5, column=0)
NFater = ttk.Entry(root)
NFater.grid(row=5, column=1)


ttk.Label(root, text="DiabetesPedigreeFunction :").grid(row=6, column=0)
NMader = ttk.Entry(root)
NMader.grid(row=6, column=1)

ttk.Label(root, text="Age :").grid(row=7, column=0)
House = ttk.Entry(root)
House.grid(row=7, column=1)

    #logo=PhotoImage(file="help.gif")
    #w=Label(root,image=logo).grid(row=0)



       
        
ttk.Button(root, text="RUN", command=predict).grid(row=9, column=7, sticky=(E, W))





    # adding some space around root window's widgets
for child in root.winfo_children():
    child.grid_configure(padx=5, pady=5)



list=[]
list.append((first_name))
print(list)

root.mainloop()

        
    

    
