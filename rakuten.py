import numpy as np
import pandas as pd
import hashlib
from sklearn.preprocessing import Imputer

class Splitter: 

    def __init__(self, list_col):
        self.list_work=list_col;
    
    def mulItem(self,row):
        return int(row['ITEM_PRICE_MIN'])*2 if row['ITEM_PRICE_MAX'] == None else row['ITEM_PRICE_MAX']

    def mulPurs(self,row):
        return int(row['PURCHASE_COUNT_MIN'])*2 if row['PURCHASE_COUNT_MAX'] == None else row['PURCHASE_COUNT_MAX']

    def mulShip(self,row):
        return int(row['SHIPPING_PRICE_MIN'])*2 if row['SHIPPING_PRICE_MAX'] == None else row['SHIPPING_PRICE_MAX']

    def transform(self, data):
        X=data.copy();
        for colname in self.list_work:
            toto= X[colname].str.split('<', 1, expand=True)
            X[colname+"_MIN"] = toto[0]
            #X[colname+"_MAX"] = toto[1]
            X[colname+"_MIN"]=X[colname+"_MIN"].str.replace('>',"")
            X.drop(colname, axis=1, inplace=True) 

        #X['ITEM_PRICE_MAX'] = X.apply(self.mulItem, axis=1)
        #X['PURCHASE_COUNT_MAX'] = X.apply(self.mulPurs, axis=1)
        #X['SHIPPING_PRICE_MAX'] = X.apply(self.mulShip, axis=1)
        #X.SHIPPING_PRICE_MAX.fillna(0, inplace=True)
        X.replace("", np.nan, inplace=True)
        X.fillna(value=0, inplace=True)
        X=X.apply(pd.to_numeric, errors='ignore')
        return X;


class DateSplitter:
    def __init__(self, colname):
        self.colname=colname;

    def transform(self, data):
        X=data.copy();
        toto= X[self.colname].str.split('/', 1, expand=True)
        X[self.colname+'_Month'] = toto[0]
        X[self.colname+'_Year'] = toto[1]
        X.drop(self.colname, axis=1, inplace=True)
        X=X.apply(pd.to_numeric, errors='ignore')
        return X;



class BoolMapper :
    mapper= {True : 1, False : 0 } # juste un dico pour dire True = 1, False = 0
    
    def __init__(self, col):
        self.col_work=col;

    def transform(self, data):
        X=data.copy();
        X[self.col_work] = X[self.col_work].map(self.mapper)
        X=X.apply(pd.to_numeric, errors='ignore')
        return X;



class WarrantCov:
    
    def __init__(self):
        pass;

    def transform(self, data):
        try:
            X=data.copy();

            #X.WARRANTIES_PRICE_MAX.fillna(value=0, inplace=True)
            X.WARRANTIES_PRICE_MIN.fillna(value=0, inplace=True)

            X.apply(pd.to_numeric, errors='ignore')
            X["WARRANTIES_COVERAGE_MIN"]=X["WARRANTIES_PRICE_MIN"].astype('float')/(X["ITEM_PRICE_MIN"].astype('float')+0.1)
            #X["WARRANTIES_COVERAGE_MAX"]=X["WARRANTIES_PRICE_MAX"].astype('float')/(X["ITEM_PRICE_MAX"].astype('float')+0.1)
            X=X.apply(pd.to_numeric, errors='ignore')
            return X;
        except AttributeError:
            print("ERROR: You must first transform the data with the Splitter class")



class Dummyzator:
    def __init__(self, target,list_col, new_value):
        self.target=target
        self.list_work=list_col;
        self.new_value=new_value

    def transform(self, data):
        X=data.copy();
        X[self.target].fillna(value="Missing_"+self.target, inplace=True)
        X[self.target].replace(to_replace=self.list_work,
                                 value=self.new_value,
                                 inplace=True)

        X = pd.merge(X, 
                   pd.get_dummies(X[self.target]), 
                   left_index=True, 
                   right_index=True)
        X.drop(self.target,
             axis=1,
             inplace=True)
        X=X.apply(pd.to_numeric, errors='ignore')
        return X;


class Stat_Adder:
    
    def __init__(self, target):
        self.target = target

    # Fonction qui prend une ligne de la BDD en entrée, et dit "Si c'est - alors on renvoit 0, sinon 1
    def b(self, row):
            return 0 if row['CLAIM_TYPE'] == '-' else 1;

    # On utilise la fonction "fit" parce qu'on doit calculer des trucs avec le jeu de train, pour pouvoir les merger dans le test
    def fit(self, data):
        # Copie de la BDD
        X=data.copy();

        # Création de la colonne binaire
        X['IS_CLAIM'] = X.apply(self.b, axis=1)

        # Calcul du group by
        self.GroupBY_DATA = X['IS_CLAIM'].groupby(X[self.target]).describe().reset_index()
        self.GroupBY_DATA.columns = [self.target,"Count_"+self.target,"Mean_"+self.target,"STD_"+self.target,"t_1","t_2","t_3","t_4","t_5"]
        self.GroupBY_DATA.drop(["t_1","t_2","t_3","t_4","t_5"], axis=1, inplace=True)
        self.GroupBY_DATA["Count_Claims_"+self.target] = self.GroupBY_DATA["Mean_"+self.target] * self.GroupBY_DATA["Count_"+self.target]


    def transform(self, data):
        X=data.copy();
        X=pd.merge(X, self.GroupBY_DATA, on=self.target, how='outer')
        X=X.apply(pd.to_numeric, errors='ignore')
        return X




class ID_Stat_Adder:
    
    def __init__(self, ID, concatenator):
        self.ID=ID;
        self.concatenator=concatenator;

    # Fonction qui prend une ligne de la BDD en entrée, et dit "Si c'est - alors on renvoit 0, sinon 1
    def b(self, row):
            return 0 if row['CLAIM_TYPE'] == '-' else 1;


    def idCreator(self,row):
        return hashlib.md5(
            (str(row[self.concatenator[0]])+
             str(row[self.concatenator[1]])+
             str(row[self.concatenator[2]])
            ).encode()
        ).hexdigest()


    def fit(self, data):
        X=data.copy();
        X['IS_CLAIM'] = X.apply(self.b, axis=1)
        X[self.ID] = X.apply(self.idCreator, axis=1)
        
        # Calcul du group by
        self.GroupBY_DATA = X['IS_CLAIM'].groupby(X[self.ID]).describe().reset_index()
        self.GroupBY_DATA.columns = [self.ID,"Count_"+self.ID,"Mean_"+self.ID,"STD_"+self.ID,"t_1","t_2","t_3","t_4","t_5"]
        self.GroupBY_DATA.drop(["t_1","t_2","t_3","t_4","t_5"], axis=1, inplace=True)
        self.GroupBY_DATA["Count_Claims_"+self.ID] = self.GroupBY_DATA["Mean_"+self.ID] * self.GroupBY_DATA["Count_"+self.ID]


    def transform(self, data):
        # Copie de la BDD
        X=data.copy();
        X[self.ID] = X.apply(self.idCreator, axis=1)
        X=pd.merge(X, self.GroupBY_DATA, on=self.ID, how='outer')
        X=X.apply(pd.to_numeric, errors='ignore')
        return X

def multiclass_roc_auc_score(truth, pred):
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(truth)
    return roc_auc_score(lb.transform(truth), lb.transform(pred), average="weighted")


def rakuten_ROC_CV(clf, X_train, Y_train, _n_splits=10, _random_state=42):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    results=[]

    skfolds = StratifiedKFold(n_splits=_n_splits, random_state=_random_state)

    for train_index, test_index in skfolds.split(X_train, Y_train):
        clone_clf = clone(clf)
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = (Y_train.iloc[train_index])
        X_test_fold = X_train.iloc[test_index]
        y_test_fold = (Y_train.iloc[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        MRA = multiclass_roc_auc_score(y_test_fold, y_pred)
        print("ROC Score: "+str(MRA))
        results.append(multiclass_roc_auc_score(y_test_fold, y_pred))

    print("___________________________")
    print("Mean ROC Score: %0.4f (+/- %0.4f)" 
              % (np.array(results).mean(), np.array(results).std()))
    return np.array(results).mean()