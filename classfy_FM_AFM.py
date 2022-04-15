from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from lightgbm import LGBMClassifier
import matplotlib.pyplot as pl
import gc
import shap
import itertools
from sklearn.metrics import confusion_matrix
import ternary
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.svm import SVC
from mlxtend.feature_selection import ColumnSelector
import mlxtend.classifier


def load_file(file_name="wuli_and_huaxue.csv"): ##读取文件，返回特征X和标签y
    df_data=pd.read_csv(file_name)
    data=np.array(df_data)
    y=data[:,2]
    X=data[:,3:]
    xx=np.zeros(shape=X.shape)
    yy=np.zeros(shape=y.shape)
    for i in range(X.shape[0]):
        for k in range(X.shape[1]):
            xx[i,k]=X[i,k]

    for i in range(yy.shape[0]):
        yy[i]=y[i]
    X=xx
    y=yy
    return df_data,X,y
def fit_lgbClassifier(X,y):##获得一个训练好的LGBMClassifier和数据集划分用于制作shap
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    data_train,data_valid, y_train, y_valid=X_train, X_test, y_train, y_test
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=30,
        colsample_bytree=.8,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
    )
    clf.fit(
        data_train, y_train, 
        eval_set= [(data_train, y_train), (data_valid, y_valid)], 
        eval_metric='auc', verbose=100, early_stopping_rounds=30  #30
    )
    return clf,data_train,data_valid, y_train, y_valid

def fit_StackingClassifier(X,y):##获得一个训练好的Stacking模型
    estimators = [('rf', RandomForestClassifier( random_state=0)),
                  ('svr', make_pipeline(StandardScaler(),
                                        LinearSVC(random_state=42)))]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(X,y)
    return clf

def plot_shap(clf,data_train,data_valid,df_data):##绘制shap图像
    source=df_data
    a=[column for column in source]
    a=a[3:]
    dt=pd.DataFrame(data_valid)
    df_new=pd.DataFrame(df_data).drop(['formula','value','type'],axis=1)
    data=df_new
    data_train=pd.DataFrame(data_train)
    data_train.columns=a
    model=clf
    X=data_train
    #explainer = shap.TreeExplainer(clf.booster_)
    explainer = shap.TreeExplainer(clf)
    shap_values=explainer.shap_values(data_train) 
    shap_values=shap_values[1]
    shap.summary_plot(shap_values, X,show = False)
    #shap.image_plot(shap_values,X, show = False)
    pl.savefig("./shap1.tif",dpi=1200,bbox_inches = "tight")
    shap.summary_plot(shap_values, X, plot_type="bar",show = False)
    pl.savefig("./shap2.tif",dpi=1200,bbox_inches = "tight")
def Oversampling(X,y):##过采样奈尔温度数据
    count=0
    for i in range(len(y)):
        if y[i]==0:
            count=count+1
    count_neer=len(y)-count
    count_juli=count
    xx=[]
    yy=[]
    for i in range(count_juli):
        xx.append(X[i,:])
        yy.append(y[i])
    
    for i in range(count_juli-count_neer):
        a=random.randint(count_juli,len(y)-1)
        xx.append(X[a,:])
        yy.append(y[a])
    for i in range(count_neer):
        xx.append(X[i+count_juli,:])
        yy.append(y[i+count_juli])
    xx=np.array(xx)
    xx=np.float32(xx)
    yy=np.array(yy)
    yy=np.float32(yy)
    return xx,yy
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=pl.cm.Blues):

    pl.imshow(cm, interpolation='nearest', cmap=cmap)
    pl.title(title,family = 'Times New Roman',fontsize = 17)
    pl.colorbar()
    tick_marks = np.arange(len(classes))
    pl.xticks(tick_marks, classes, rotation=0)
    pl.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pl.text(j, i, cm[i, j],fontsize =17,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pl.tight_layout()
    pl.ylabel('True label',fontsize =15,family = 'Times New Roman')
    pl.xlabel('Predicted label',fontsize =15,family = 'Times New Roman')
    
def plot_matrix(clf,Xtest,Ytest):##绘制混淆矩阵
    prodict_prob_y=clf.predict_proba(Xtest)[:,1]
    y_probabilities_rf=prodict_prob_y
    thresholds = [0.5]
    pl.figure()
    m = 1
    i = 0.5
    y_test_predictions_high_recall = y_probabilities_rf > i    
    y_test=Ytest    
    cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
    print(cnf_matrix)
    np.set_printoptions(precision=2)  
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix,classes=class_names)
def cost_sensitive(clf,Xtest,Ytest):##绘制代价敏感曲线
    prodict_prob_y=clf.predict_proba(Xtest)[:,1]
    y_probabilities_rf=prodict_prob_y
    lll = []
    for i in range(100,900,1):
        lll.append(i/1000)
    mingan = []
    thresholds = lll
    m = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_probabilities_rf > i  
        #cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
        cnf_matrix = confusion_matrix(Ytest,y_test_predictions_high_recall)
        np.set_printoptions(precision=2)
        C01 = cnf_matrix[0][1]
        C10 = cnf_matrix[1][0]
        C00 = cnf_matrix[0][0]
        C11 = cnf_matrix[1][1]
        C =C01+C00+C10+C11
        cost = (2*(C01)/((C11+C10)))+((1.5*C10)/((C11+C01)))
        COST=cost
        mingan.append(COST)
    
    mingan = np.array(mingan)
    lll = np.array(lll)
    pl.plot(lll,mingan,label='cost value')
    pl.scatter(lll,mingan,c="r",s = 12,alpha=0.7)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 15,
    }
    pl.ylabel('cost',font2)
    pl.xlabel(' Threshold value',font2)

    fig = pl.gcf( )
    fig.set_size_inches(8, 6)

    pl.legend()
    pl.savefig('daijia_cost.jpg', dpi=300,bbox_inches ="tight")
    pl.show( )
    mingan = np.array(mingan)
    lll = np.array(lll)
    pl.plot(lll,mingan)
    pl.scatter(lll,mingan,c="r",s = 12,alpha=0.7)
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 15,
    }

    fig = pl.gcf( )
    pl.xlim(0.5,0.7)
    pl.ylim(0.15,0.3)
    fig.set_size_inches(8, 6)
    pl.show( ) 
def plot_three_phase():   ##绘制三元相图
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 15,
    'color':'r',
    }
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    str_title='$'+element1+'_x '+element2+'_y '+element3+'_z$'
    #str1=""+element1+"---"+element2
    str1=""+element1
    #str2=""+element1+"---"+element3
    str3=""+element3
    #str3=""+element2+"---"+element3
    str2=""+element2
    #tax.set_title(r'$H_x Cr_y O_z$')
    #tax.set_title(str_title)
    tax.heatmapf(shannon_entropy, boundary=True, style="triangular",cbarlabel='probility of FM',vmax=1.0, vmin=0.0)
    tax.boundary(linewidth=2.0)
    tax.ticks(axis='lbr',linewidth=1,multiple=25,fontsize=14,offset=0.02)
    #tax.annotate(element1,(-2,0,100),fontsize=14)
    #tax.annotate(element2,(-5,100,100),fontsize=14)
    #tax.annotate(element3,(100,2,100),fontsize=14)
    tax.left_axis_label(str1,offset=0.16)
    tax.bottom_axis_label(str2,offset=0.16)
    tax.right_axis_label(str3,offset=0.16)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    tax.show()
    tax.savefig("ex.png",dpi=1200,bbox_inches = "tight")
def shannon_entropy(p):##得到三元相图中每个点是铁磁的概率
    """Computes the Shannon Entropy at a distribution in the simplex."""
    a=np.zeros(shape=(1,ele_num))
    p=np.array(p)
    p=np.float32(p)
    a[0][ele1]=p[0]
    a[0][ele2]=p[1]
    a[0][ele3]=p[2]
    return clf_phase.predict_proba(a)[0,0]
def model_sele():
    df_data,X,y=load_file()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    xx,yy=Oversampling(X_train,y_train)

    wuli=()
    huaxue=()
    for i in range(120):
        wuli=(xx.shape[1]-i-1,)+wuli
    for i in range(88):
        huaxue=huaxue+(i,)
    pipe1 = make_pipeline(ColumnSelector(cols=wuli),
                         ExtraTreesClassifier())
    pipe2 = make_pipeline(ColumnSelector(cols=wuli),
                          RandomForestClassifier())

    pipe3 = make_pipeline(ColumnSelector(cols=huaxue),
                         ExtraTreesClassifier())
    pipe4 = make_pipeline(ColumnSelector(cols=huaxue),
                          RandomForestClassifier())
    sclf = mlxtend.classifier.StackingClassifier(classifiers=[pipe1, pipe2,pipe3,pipe4], 
                                  meta_classifier=LogisticRegression())
    clfs=[RandomForestClassifier(),xgb.XGBClassifier(),ExtraTreesClassifier(),KNeighborsClassifier(),sclf]
    acc=['acc']
    recall=['recall']
    auc=['auc']
    FM_pre=['FM_pre']
    for clf in clfs:
        clf.fit(xx,yy)
        prodict_prob_y=clf.predict_proba(X_test)[:,1]
        y_probabilities_rf=prodict_prob_y
        y_test_predictions_high_recall = y_probabilities_rf > 0.5
        cnf_matrix = confusion_matrix(y_test,y_test_predictions_high_recall)
        FM_pre.append(cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1]))
        acc.append(clf.score(X_test,y_test))
        recall.append(metrics.recall_score(y_test,clf.predict(X_test)))
        auc.append(metrics.roc_auc_score(y_test,prodict_prob_y))
        #print(clf.score(X_test,y_test))
    point=[acc,recall,auc,FM_pre]
    a=pd.DataFrame(point,columns=['','RF','xgb','ETC','knn','stacking'])
    a.to_csv("point.csv")
    print("resoult save in point.csv")
    return a 



    
    

df_data,X,y=load_file()
a=df_data.columns
for i in range(len(a)):
    if a[i]=='Ra':
        break
ele_num=i+1-3
X=X[:,0:ele_num]
xx,yy=Oversampling(X,y)
X=xx
y=yy
clf_phase=fit_StackingClassifier(X,y)
element1=''
element2=''
element3=''
ele1=0
ele2=0
ele3=0
command_str="\nplease a command number \n0:exit application\n1:plot shap\n2:plot confusion_matrix\n3:plot cost sensitive\n4:plot 3 phase \n5:model selection\n->"
while  (1):
    cmd=input(command_str)
    if cmd=='1':
        df_data,X,y=load_file()
        clf,data_train,data_valid, y_train, y_valid=fit_lgbClassifier(X,y)
        plot_shap(clf,data_train,data_valid,df_data)
    if cmd=='2':
        df_data,X,y=load_file()
        X,y=Oversampling(X,y)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, stratify=y, random_state=42)
        clf=fit_StackingClassifier(Xtrain,Ytrain)
        plot_matrix(clf,Xtest,Ytest)
    if cmd=='3':
        df_data,X,y=load_file()
        X,y=Oversampling(X,y)
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, stratify=y, random_state=42)
        clf=fit_StackingClassifier(Xtrain,Ytrain)
        cost_sensitive(clf,Xtest,Ytest)
    if cmd =='4':
        element1=input("first element\n->")
        element2=input("second element\n->")
        element3=input("third element\n->")
        for i in range(ele_num):
            if a[3+i]==element1:
                ele1=i
                break
        for i in range(ele_num):
            if a[3+i]==element2:
                ele2=i
                break
        for i in range(ele_num):
            if a[3+i]==element3:
                ele3=i
                break
        plot_three_phase()
    if cmd =='5':
        model_sele()
        break
        
    if cmd=='0':
        print("process end\nthank you")
        break
    
