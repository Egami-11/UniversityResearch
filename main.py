import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import datetime 
from tqdm import tqdm#プログレスバー
import csv
from sklearn import preprocessing#正規化

#データセット
from sklearn.datasets import load_iris

#クラスタリング手法
from sklearn.cluster import KMeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans

#評価手法
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score #調整ランド指標ARI
from sklearn.metrics.cluster import normalized_mutual_info_score #正規化相互情報量NMI
import scipy.spatial#fcm_get_uメソッドで使用

#Rから呼び出すパッケージ
import pyper
r=pyper.R()
r("library(clValid)")#dunn-index
r("library(clusterCrit)")#PBM XBなどの評価手法
r("library(fclust)")#FKM ファージーkmeans法
r("library(frbs)")#ECMのパッケージ


def setup():
    #PythonからRを呼び出す際，最初の1回目の呼び出しがうまく実行されない可能性があるため、失敗しても続行させるために再度実行する
    iris=load_iris()
    r.assign("data",iris.data)
    r.assign("labels",iris.target)
    r("ans <- intCriteria(data,labels,c('Dunn'))")
    ans=r.get("ans")

#データセットの読み込み及びデータクレンジング
class DataSelect:
    def __init__(self,data_name,normaliz=True,standardiz=False,shuffle=True):
        self.data_cleansing={}
        self.data_cleansing['normaliz']=normaliz
        self.data_cleansing['standardiz']=standardiz
        self.data_cleansing['shuffle']=shuffle
        
        directory_name="./dataset/"
        
        if data_name=='iris':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=151-1#全行数-1
            DD=5#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1
        elif data_name=='wine':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=179-1#全行数-1
            DD=14#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1
        elif data_name=='breast-cancer':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=570-1#全行数-1
            DD=31#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1
        elif data_name=='glass':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=215-1#全行数-1
            DD=10#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    if int(row[DD-1])<=3:base_target[i-1]=int(row[DD-1])-1
                    else:base_target[i-1]=int(row[DD-1])-2
                i+=1
        elif data_name=='thyroid':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=216-1#全行数-1
            DD=6#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])-1
                i+=1
        elif data_name=='user-knowledge-modeling':
            self.data_cleansing['normaliz']=False
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=259-1#全行数-1
            DD=6#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1
        elif data_name=='raisin':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=901-1#全行数-1
            DD=8#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1
        elif data_name=='seeds':
            file_name=directory_name+data_name+".csv"
            csv_file = open(file_name, "r", encoding="ms932", errors="", newline="" )
            f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
            NN=211-1#全行数-1
            DD=8#=特徴数+ラベル
            base_data=np.zeros(NN*(DD-1)).reshape(NN,DD-1)#.astype(dtype=np.int16)
            base_target=np.zeros(NN).astype(dtype=np.int16)
            i=0
            for row in f:
                if i==0:
                    self.feature_name=row[:-1]
                else:
                    for j in range(DD-1):
                        base_data[i-1,j]=row[j]
                    base_target[i-1]=int(row[DD-1])
                i+=1

        #正規化
        if self.data_cleansing['normaliz']:
            mm = preprocessing.MinMaxScaler()
            base_data = mm.fit_transform(base_data)

        #標準化
        if self.data_cleansing['standardiz']:base_data=self.standardization(base_data)
        
        shuffle_sele=list(range(base_data.shape[0]))
        np.random.seed(0)#乱数シードを固定(実験結果を残しておきたいとき用)
        np.random.shuffle(shuffle_sele)#多次元配列の場合random.shuffleだとデータが壊れるためnumpyの方を使う
        if self.data_cleansing['shuffle']:#データをシャッフルする
            self.data=base_data[shuffle_sele,]
            self.target=base_target[shuffle_sele,]
        else:
            self.data=base_data
            self.target=base_target
        self.base_data=base_data
        self.base_target=base_target
        self.shuffle_sele=shuffle_sele
        self.data_name=data_name

    def standardization(self,x, axis=None, ddof=0):#標準化
        x_mean = x.mean(axis=axis, keepdims=True)
        x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
        return (x - x_mean) / x_std
    
    @staticmethod
    #シャッフル状態を元の状態に戻す
    def reset_shuffle(array, seed=0):
        if array.ndim == 2:#2次元配列の場合
            array2d=[]
            for i in range(array.shape[1]):
                seq = np.arange(len(array[:,i].T))
                np.random.seed(seed)
                np.random.shuffle(seq)
                tmp = np.c_[seq.T, np.array(array[:,i].T).T]
                tmp = np.ndarray.tolist(tmp)
                tmp = sorted(tmp)
                tmp = np.array(tmp)
                array2d.append(np.ndarray.tolist(tmp[:,1]))
            return np.array(array2d).T
        elif array.ndim == 1:
            seq = np.arange(len(array))
            np.random.seed(seed)
            np.random.shuffle(seq)
            tmp = np.c_[seq.T, np.array(array).T]
            tmp = np.ndarray.tolist(tmp)
            tmp = sorted(tmp)
            tmp = np.array(tmp)
            tmp = np.ndarray.tolist(tmp[:,1])
            return np.array(tmp)

class Validation:
    evaluation_name="ch dunn pbm sil db xb".split()
    targets = "max max max max min min".split()
    
    def __init__(self, data, labels, centers, m=2.0, reciprocal=True):
        self.data = data
        self.labels = labels
        self.centers = centers
        self.m = m
        self.reciprocal = reciprocal#True:逆数
        self.evaluation_value = np.nan#評価値
        self.methods = [self.ch, self.dunn, self.pbm, self.sil, self.db, self.xb]
        self.evaluation_list={}#最良評価値を記録
        for eva_name in Validation.evaluation_name:
            self.evaluation_list[eva_name]=0.0
            
    def evaluation(self):
        for method,eva_name in zip(self.methods,Validation.evaluation_name):
            self.evaluation_list[eva_name]=method()
        return self.evaluation_list
    
    def evaluation_non_xb(self):#スペクトラルクラスタリング用（クラスタセンターが計算ができないため）
        for method,eva_name in zip(self.methods,Validation.evaluation_name):
            if eva_name != 'xb':self.evaluation_list[eva_name]=method()
        return self.evaluation_list
    
    def validation_metrics_available(self):
        #import re
        methods =  [method for method in dir(self) if callable(getattr(self, method))]
        methods.remove('validation_metrics_available')#除外
        methodDict = {}
        for method in methods:
            if not re.match('__', method) and not re.match('_validation__', method):
                methodDict[method] = ''
        return methodDict
    
    def ch(self):
        if 2 <= len(cluster_size(self.labels)):
            self.evaluation_value=calinski_harabasz_score(self.data, self.labels)
        else:
            self.evaluation_value=0.0
        return self.evaluation_value
    
    def dunn(self):
        r.assign("data",self.data)
        r.assign("labels",self.labels)
        r("ans <- intCriteria(data,labels,c('Dunn'))")
        ans=r.get("ans")
        
        if ans == None:
            self.evaluation_value=0.0
        else:
            self.evaluation_value=ans["dunn"]
        return self.evaluation_value
    
    def pbm(self):
        r.assign("data",self.data)
        r.assign("labels",self.labels)
        r("ans <- intCriteria(data,labels,c('PBM'))")
        ans=r.get("ans")
        if ans == None:
            self.evaluation_value=0.0
        else:
            self.evaluation_value=ans["pbm"]
        return self.evaluation_value
    
    def sil(self):
        if 2 <= len(cluster_size(self.labels)):
            self.evaluation_value=silhouette_score(self.data, self.labels)
        else:
            self.evaluation_value=0.0
        return self.evaluation_value
    
    def db(self):
        if 2 <= len(cluster_size(self.labels)):
            db=davies_bouldin_score(self.data, self.labels)
            if db==0.0:self.evaluation_value=0.0
            elif self.reciprocal: self.evaluation_value=1/db
            else : self.evaluation_value=db
        else:
            self.evaluation_value=0.0
        return self.evaluation_value

    def xb(self):
        if not(type(self.centers) is float):
            u = self.fcm_get_u()
            r.assign("data",self.data)
            r.assign("u",u.T)
            r.assign("c",self.centers)
            r.assign("m",self.m)
            r("xb=XB(data,u,c,m)")
            ans=r.get("xb")
            if self.reciprocal: self.evaluation_value=1/ans
            else : self.evaluation_value=ans
        else:self.evaluation_value=0.0
        return self.evaluation_value
    
    def fcm_get_u(self):#,x, v, m):# u = fcm_get_u(x, v, m),v=クラスタ中心位置
        distances = scipy.spatial.distance.cdist(self.data, self.centers)**2#距離行列を作る
        #nonzero_distancesで最長距離(fmax:nanを除く)を求める
        nonzero_distances = np.fmax(distances, np.finfo(np.float64).eps)#finfo:float64bitの指数部(eps)の取り扱える最大最小範囲のビット数を取得
        inv_distances = np.reciprocal(nonzero_distances)**(1/(self.m - 1))#reciprocal:逆数を求める
        return inv_distances.T/np.sum(inv_distances, axis=1)
    
    def sse(self):#クラスタ内誤差平方和
        sse=0.0
        for i in range(len(self.data)):sse += np.sum((self.data[i] - self.centers[self.labels[i]])**2)
        return sse

class Clustering:
    def __init__(self,data,dthr,min_samples=10):
        self.data=data
        self.dthr=dthr
        self.min_samples=min_samples
        self.labels=np.zeros(data.shape[0]).astype(dtype=np.int16)
        
    def run(self):
        self.ecm()
        
    def ecm(self):
        cluster_centers = {} #クラスタ中心位置
        cluster_radii = {} #クラスタ半径
        num_of_cluster_centers = 0 #クラスタ番号
        cluster_r=[]#クラスタ割り振り状態
        for i in np.arange(len(self.data)):
            if not cluster_centers: #クラスタに何もデータがない場合(クラスタ初期設定)
                cluster_centers[num_of_cluster_centers] = self.data[i]
                cluster_radii[num_of_cluster_centers] = 0.0
                cluster_r.append(num_of_cluster_centers)
                num_of_cluster_centers += 1
            else:
                min_dist_and_cluster = self.minimal_dist_from_centers(self.data[i], cluster_centers)
                if not self.check(min_dist_and_cluster[0], cluster_radii.values()):#データとクラスタ中心の距離が、クラスタ半径内に収まらない場合
                    min_dist_from_extended = self.minimal_dist_from_extended_values(self.data[i], cluster_centers, cluster_radii)#=S_a
                    if min_dist_from_extended[0] > (2.0 * self.dthr):#[閾値外]新たなクラスタを作成
                        cluster_centers[num_of_cluster_centers] = self.data[i]
                        cluster_radii[num_of_cluster_centers] = 0.0
                        cluster_r.append(num_of_cluster_centers)
                        num_of_cluster_centers += 1
                    else:#[閾値内]既存のクラスタへ割当
                        cluster_center = cluster_centers[min_dist_from_extended[1]]#S_aのクラスタ
                        new_radius = min_dist_from_extended[0] / 2.0 
                        temp = self.dd(self.data[i], cluster_center)#dd:[0]ユークリッド距離:[1]X-Cc[2]ここだけ正規ではない
                        ratio = np.abs(temp[0] - new_radius) / temp[0]
                        new = ratio * temp[1]
                        cluster_centers[min_dist_from_extended[1]] = new + cluster_center
                        cluster_radii[min_dist_from_extended[1]] = new_radius
                        cluster_r.append(min_dist_and_cluster[2])
                else:cluster_r.append(min_dist_and_cluster[2])
        result=[]
        for i in cluster_centers:result.append(list(cluster_centers[i]))
        self.centers=np.array(result)
        self.labels=np.array(cluster_r)
        return self.labels,self.centers#1:クラスタラベル、2:クラスタ中心位置
    
    def d(self,i, j):#正規化ユークリッド距離
        temp = i - j
        return np.sqrt(np.sum(np.power(temp, 2))) / np.sqrt(len(i))

    def dd(self,i,j):#標準のユークリッド距離
        temp = i - j #X-Center
        return [np.sqrt(np.sum(np.power(temp, 2))), temp]#,X-Cc
    
    def minimal_dist_from_centers(self,i, cluster_centers):
        dists = []
        for j in cluster_centers.values():
            dists.append(self.d(i, j))
        return [np.min(dists), np.argmin(dists),np.argmin(dists)] #クラスタ間距離が最短の値とそのクラスタの番号を返す
    
    def minimal_dist_from_extended_values(self,i, cluster_centers, cluster_radii):#S_aを求める関数
        dists = []
        for j, radius_j in zip(cluster_centers.values(), cluster_radii.values()):#
            dists.append(self.d(i, j) + radius_j)
        return [np.min(dists), np.argmin(dists)]
    
    def check(self,d, cluster_radii): 
        return any(d <= np.array(list(cluster_radii)))

def cluster_centers_xb(data,labels):#各クラスタの中心点(XB用：XBでは他の使用した評価手法と違い，評価する際にクラスタ中心情報が必要となるため)    
    cl_size=cluster_size(labels)
    cl_num=len(cl_size)
    n=data.shape[0]
    d=data.shape[1]
    cluster_centers=np.zeros(cl_num*d).reshape(cl_num,d)#クラスタ中心位置

    for clus_num in range(cl_num):
        buf=np.zeros(d)
        num=0
        for i in np.arange(n):
            if labels[i]==clus_num:
                buf += data[i]
                num+=1
        cluster_centers[clus_num] = [val / num for val in buf]
        
    return cluster_centers

class ABC:
    def __init__(self,eva_name,data,target,food_num=10,feature_num_min=2,cluster_num_max=10,sol_lim_min=0.0,sol_lim_max=1.0,feature_sele_dthr=0.5,mabc_run=False,mabc_mr=0.5,ecm_dthr_min=1.0e-05,ecm_dthr_max=1.0):
        n=data.shape[0]
        d=data.shape[1]
        self.eva_name=eva_name
        self.best_evaluation={}
        self.best_feature=np.zeros(d).astype(dtype=np.int16)
        self.best_dthr=0.0
        self.best_clus_size = {}
        self.best_labels=np.zeros(n).astype(dtype=np.int16) 
        self.best_centers=0
        self.all_eva_name=Validation.evaluation_name
        for eva in self.all_eva_name:self.best_evaluation[eva]=0.0
        
        self.abc_methods = "init employed onlooker scout".split()
        self.update_fit={}
        for method in self.abc_methods:self.update_fit[method]=0
        
        self.target=target
        self.foods=np.zeros(food_num*d).reshape(food_num,d)
        self.feature=np.zeros(food_num*d).reshape(food_num,d).astype(dtype=np.int16)
        self.dthr=np.zeros(food_num)
        self.fitness=np.zeros(food_num)
        self.trial=np.zeros(food_num)
        self.prob=np.zeros(food_num)
        #定数
        self.food_num=food_num#個体数
        self.cycle_num=0
        self.n=n
        self.d=d
        self.data=data
        self.sol_lim_min=sol_lim_min
        self.sol_lim_max=sol_lim_max
        self.feature_sele_dthr=feature_sele_dthr#この値以上ならその特徴を使用
        self.ecm_dthr_min=ecm_dthr_min #乱数生成の下限値
        self.ecm_dthr_max=ecm_dthr_max #乱数生成の上限値
        self.scout_limit=food_num*d
        self.mabc_run=mabc_run #MABCの使用可否
        self.mabc_mr=mabc_mr #修正率MR [乱数 < MRのときその解を更新] 
        self.feature_num_min=feature_num_min #特徴をこの値以上に選択されるまで繰り返す
        self.cluster_num_max=cluster_num_max #クラスタ数を制限し、この数以上の結果は残さないでスキップする。
        
    def run(self,cycle_num=1000):
        self.cycle_num=cycle_num
        self.cycle_fit=[]#繰り返し回数に対する評価値
        self.initial()
        for cycle in tqdm(range(1,self.cycle_num+1)):
            self.employed_bee()
            self.onlooker_bee()
            self.scout_bee()
            self.cycle_fit.append(float("{:.3f}".format(self.best_evaluation[self.eva_name])))
            
        for c in np.unique(self.best_labels):self.best_clus_size[c] = np.count_nonzero(self.best_labels == c)
        
        return self.best_evaluation,self.best_feature,self.best_dthr,self.best_clus_size,self.best_labels

    def initial(self):
        f_s=np.zeros(self.food_num*self.d).reshape(self.food_num,self.d).astype(dtype=np.int16)
        for i in range(self.food_num):
            while True:
                for j in range(self.d):
                    #np.random.seed(0)#乱数シードを固定(実験結果を残しておきたいとき用)
                    self.foods[i,j]=np.random.rand()*(self.sol_lim_max-self.sol_lim_min)+self.sol_lim_min
                    if self.sol_lim_max < self.foods[i,j]:self.foods[i,j]=self.sol_lim_max
                    elif self.foods[i,j] < self.sol_lim_min:self.foods[i,j]=self.sol_lim_min
                f_s[i]=self.feature_selection(self.foods[i])
                if self.feature_num_min<=sum(f_s[i]):break#特徴を1つ以上選択されるまでループ

            self.dthr[i]=np.random.rand()*(self.ecm_dthr_max-self.ecm_dthr_min)+self.ecm_dthr_min
            if self.dthr[i] < self.ecm_dthr_min:self.dthr[i]=self.ecm_dthr_min
            elif self.ecm_dthr_max < self.dthr[i]:self.dthr[i]=self.ecm_dthr_max
            
            select_data=self.data[:,f_s[i]==1]
            ecm=Clustering(select_data,self.dthr[i])
            ecm.run()
            labels=ecm.labels
            n_clust_now=max(labels)#現在のクラスタ数
            if n_clust_now == 0:self.fitness[i]=1.0E-20#クラスタ数が１つだけのとき適応度を最小に設定
            elif n_clust_now<=self.cluster_num_max-1:
                eva=Validation(select_data,labels,ecm.centers)
                eva.evaluation()
                for n_eva,name in enumerate(self.all_eva_name):
                    if self.eva_name==name:
                        self.fitness[i]=eva.evaluation_list[name]
                        self.update_fit['init']+=1
                        self.trial[i] = 0

                        if self.best_evaluation[name] < self.fitness[i]:
                            for a_eva_name in self.all_eva_name:
                                self.best_evaluation[a_eva_name]=eva.evaluation_list[a_eva_name]
                            self.best_feature=f_s[i]
                            self.best_labels=labels
                            self.best_dthr=self.dthr[i]
                            self.best_centers=ecm.centers
                    else:
                        self.trial[i] = self.trial[i] + 1
            else:
                self.trial[i] = self.trial[i] + 1

    def employed_bee(self):
        f_s=np.zeros(self.food_num*self.d).reshape(self.food_num,self.d).astype(dtype=np.int16)
        for i in range(self.food_num):
            solution=copy.deepcopy(self.foods[i])
            f_s[i]=self.feature_selection(solution)
            dthr=self.dthr[i]
            j=0
            change_parameter=0
            if self.mabc_run:#MABC
                for j in range(self.d):
                    if np.random.rand() < self.mabc_mr:
                        change_parameter=1
                        solution[j]=self.solution_update(i,j)
                if np.random.rand() < self.mabc_mr:
                    change_parameter=1
                    dthr=self.dthr_update(i)
                f_s[i]=self.feature_selection(solution)
            if self.mabc_run==False or change_parameter==0 or sum(f_s[i]) < self.feature_num_min:#ABC
                solution=copy.deepcopy(self.foods[i])
                if self.feature_sele_dthr==0.0:j=self.d
                elif sum(f_s[i]) < self.feature_num_min:j=np.random.randint(0,self.d)
                else:j=np.random.randint(0,self.d+1)

                if j==self.d:#閾値更新
                    dthr=self.dthr_update(i)
                else:#特徴選択
                    while True:
                        solution[j]=self.solution_update(i,j)
                        f_s[i]=self.feature_selection(solution)
                        if self.feature_num_min <= sum(f_s[i]):break

            select_data=self.data[:,f_s[i]==1]
            ecm=Clustering(select_data,dthr)
            ecm.run()
            labels=ecm.labels
            n_clust_now=max(labels)#現在のクラスタ数
            if n_clust_now == 0:self.fitness[i]=1.0E-20#クラスタ数が１つだけのとき適応度を最小に設定
            elif n_clust_now<=self.cluster_num_max-1:
                eva=Validation(select_data,labels,ecm.centers)
                eva.evaluation()
                for n_eva,name in enumerate(self.all_eva_name):
                    if self.eva_name==name and self.fitness[i] < eva.evaluation_list[name]:
                        self.fitness[i]=eva.evaluation_list[name]
                        self.foods[i]=solution
                        self.dthr[i]= dthr
                        self.update_fit['employed']+=1
                        self.trial[i] = 0

                        if self.best_evaluation[name] < self.fitness[i]:
                            for a_eva_name in self.all_eva_name:
                                self.best_evaluation[a_eva_name]=eva.evaluation_list[a_eva_name]
                            self.best_feature=f_s[i]
                            self.best_labels=labels
                            self.best_dthr=self.dthr[i]
                            self.best_centers=ecm.centers
                    else:
                        self.trial[i] = self.trial[i] + 1
            else:
                self.trial[i] = self.trial[i] + 1
                

    def onlooker_bee(self):
        for i in range(self.food_num):
            self.prob[i] = self.fitness[i] / self.fitness.sum()

        f_s=np.zeros(self.food_num*self.d).reshape(self.food_num,self.d).astype(dtype=np.int16)

        for _ in range(self.food_num):
            r=np.random.uniform(0,1)
            m=0.0
            for n in range(self.food_num):
                if m <= r and r <= m + self.prob[n]:
                    i=n
                    break
                m+=self.prob[n]
            solution=copy.deepcopy(self.foods[i])
            f_s[i]=self.feature_selection(solution)
            dthr=self.dthr[i]
            j=0
            change_parameter=0
            if self.mabc_run:#MABC
                for j in range(self.d):
                    if np.random.rand() < self.mabc_mr:
                        change_parameter=1
                        solution[j]=self.solution_update(i,j)
                if np.random.rand() < self.mabc_mr:
                    change_parameter=1
                    dthr=self.dthr_update(i)
                f_s[i]=self.feature_selection(solution)
            if self.mabc_run==False or change_parameter==0 or sum(f_s[i]) < self.feature_num_min:#ABC
                solution=copy.deepcopy(self.foods[i])
                if self.feature_sele_dthr==0.0:j=self.d
                elif sum(f_s[i]) < self.feature_num_min:j=np.random.randint(0,self.d)
                else:j=np.random.randint(0,self.d+1)

                if j==self.d:#閾値更新
                    dthr=self.dthr_update(i)
                else:#特徴選択
                    while True:
                        solution[j]=self.solution_update(i,j)
                        f_s[i]=self.feature_selection(solution)
                        if self.feature_num_min <= sum(f_s[i]):break

            select_data=self.data[:,f_s[i]==1]
            ecm=Clustering(select_data,dthr)
            ecm.run()
            labels=ecm.labels
            n_clust_now=max(labels)#現在のクラスタ数
            if n_clust_now == 0:self.fitness[i]=1.0E-20#クラスタ数が１つだけのとき適応度を最小に設定
            elif n_clust_now<=self.cluster_num_max-1:
                eva=Validation(select_data,labels,ecm.centers)
                eva.evaluation()
                for n_eva,name in enumerate(self.all_eva_name):
                    if self.eva_name==name and self.fitness[i] < eva.evaluation_list[name]:
                        self.fitness[i]=eva.evaluation_list[name]
                        self.foods[i]=solution
                        self.dthr[i]= dthr
                        self.update_fit['onlooker']+=1
                        self.trial[i] = 0

                        if self.best_evaluation[name] < self.fitness[i]:
                            for a_eva_name in self.all_eva_name:
                                self.best_evaluation[a_eva_name]=eva.evaluation_list[a_eva_name]
                            self.best_feature=f_s[i]
                            self.best_labels=labels
                            self.best_dthr=self.dthr[i]
                            self.best_centers=ecm.centers
                    else:
                        self.trial[i] = self.trial[i] + 1
            else:
                self.trial[i] = self.trial[i] + 1

    def scout_bee(self):
        f_s=np.zeros(self.food_num*self.d).reshape(self.food_num,self.d).astype(dtype=np.int16)
        if self.trial.max() >= self.scout_limit:
            i=self.trial.argmax()
            while True:
                for j in range(self.d):
                    self.foods[i,j]=np.random.rand()*(self.sol_lim_max-self.sol_lim_min)+self.sol_lim_min
                    if self.sol_lim_max < self.foods[i,j]:self.foods[i,j]=self.sol_lim_max
                    elif self.foods[i,j] < self.sol_lim_min:self.foods[i,j]=self.sol_lim_min
                f_s[i]=self.feature_selection(self.foods[i])
                if self.feature_num_min<=sum(f_s[i]):break#特徴を2つ以上選択されるまでループ
            select_data=self.data[:,f_s[i]==1]

            self.dthr[i]=np.random.rand()*(self.ecm_dthr_max-self.ecm_dthr_min)+self.ecm_dthr_min
            #if self.dthr[i] == 0.0:self.dthr[i]=self.ecm_dthr_min
            if self.dthr[i] < self.ecm_dthr_min:self.dthr[i]=self.ecm_dthr_min
            elif self.ecm_dthr_max < self.dthr[i]:self.dthr[i]=self.ecm_dthr_max

            ecm=Clustering(select_data,self.dthr[i])
            ecm.run()
            labels=ecm.labels
            n_clust_now=max(labels)#現在のクラスタ数
            if n_clust_now == 0:self.fitness[i]=1.0E-20#クラスタ数が１つだけのとき適応度を最小に設定
            elif n_clust_now<=self.cluster_num_max-1:
                eva=Validation(select_data,labels,ecm.centers)
                eva.evaluation()
                for n_eva,name in enumerate(self.all_eva_name):
                    if self.eva_name==name:
                        self.fitness[i]=eva.evaluation_list[name]
                        self.update_fit['scout']+=1
                        self.trial[i] = 0

                        if self.best_evaluation[name] < self.fitness[i]:
                            for a_eva_name in self.all_eva_name:
                                self.best_evaluation[a_eva_name]=eva.evaluation_list[a_eva_name]
                            self.best_feature=f_s[i]
                            self.best_labels=labels
                            self.best_dthr=self.dthr[i]
                            self.best_centers=ecm.centers

    def feature_selection(self,foods):
        fs=[]
        for i in range(len(foods)):
            if self.feature_sele_dthr<=foods[i]:fs.append(1)
            else:fs.append(0)
        return fs
    
    def solution_update(self,i,j):
        phi=np.random.uniform(-1,1)
        k=0
        while True:
            k=np.random.randint(0,self.food_num)
            if(k != i):break
        sol=self.foods[i,j]+phi*(self.foods[i,j]-self.foods[k,j])
        if sol < self.sol_lim_min:sol = self.sol_lim_min
        elif sol > self.sol_lim_max:sol = self.sol_lim_max
        return sol
    
    def dthr_update(self,i):
        phi=np.random.uniform(-1,1)
        k=0
        while True:
            k=np.random.randint(0,self.food_num)
            if(k != i):break
        dthr=self.dthr[i]+phi*(self.dthr[i]-self.dthr[k])
        #if dthr == 0.0:dthr=self.ecm_dthr_min
        if dthr < self.ecm_dthr_min:dthr=self.ecm_dthr_min
        elif self.ecm_dthr_max < dthr:dthr=self.ecm_dthr_max
        return dthr

def cluster_size(labels):
    cl_size = {}
    for c in np.unique(labels):
        cl_size[c] = np.count_nonzero(labels == c)
    return cl_size

#実行結果をファイルに出力
class OutputResult():
    def __init__(self,prog_name,data_name,data_cleansing,target,shuffle_sele,feature_name):
        self.shuffle_sele=shuffle_sele
        self.data_cleansing=data_cleansing
        self.target=target
        self.data_name=data_name
        self.run_time=datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
        self.file_name_base='_'+prog_name+'_'+data_name+'_save'+datetime.datetime.today().strftime("%Y-%m-%d")
        self.directory_name='実行結果\\\\'+prog_name
        os.makedirs(self.directory_name, exist_ok=True)#ディレクトリの新規作成
        self.file_name=os.path.join(self.directory_name, self.file_name_base+'.txt')
        self.file_name_all=os.path.join(self.directory_name, self.file_name_base+'_詳細.txt')
        self.feature_name=np.array(feature_name)

    def save(self,result):
        keys=result.keys()
        best_feature=result['best_feature']
        best_evaluation=result['best_evaluation']
        best_centers=result['best_centers']
        all_eva_name=result['all_eva_name']
        eva_name=result['eva_name']
        cluster_num_max=result['cluster_num_max']
        self.data=result['data']
        self.best_labels=result['best_labels']
        self.best_clus_size=result['best_clus_size']
        self.target_size=cluster_size(self.target)
        self.target_num=len(self.target_size)    
        self.eva_name=eva_name
        self.cycle_num=result['cycle_num']
        self.cycle_fit=result['cycle_fit']
        self.cycle_fit_fig()
        self.plot_data(self.data[:,best_feature==1],self.best_labels,best_centers,self.feature_name[best_feature==1],'ECM')

        now_t=datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
            
        if sum(best_feature)!=0:        
            #<Baseline>
            base_eva={}
            ecm=Clustering(self.data[:,best_feature==1],0.0)
            base_centers=cluster_centers_xb(self.data[:,best_feature==1],self.target)
            base=Validation(self.data[:,best_feature==1],self.target,base_centers)
            base.evaluation()
            for name in all_eva_name:
                base_eva[name]=base.evaluation_list[name]
            self.plot_data(self.data[:,best_feature==1],self.target,base_centers,self.feature_name[best_feature==1],'BASE')
                
            #<K-means>
            k_m_eva={}
            km_sse=[]
            k_m_clus_labels=[]
            k_m_clus_size=[]
            for name in all_eva_name:
                k_m_eva[name]=[]
            
            for cs in range(cluster_num_max-1):
                model = KMeans(n_clusters=cs+2).fit(self.data[:,best_feature==1])#model.labels_ model.cluster_centers_
                cl_size = {}
                k_m_clus_labels.append(model.labels_)
                for c in np.unique(k_m_clus_labels[cs]):cl_size[c] = np.count_nonzero(k_m_clus_labels[cs] == c)
                k_m_clus_size.append(cl_size)
                km_labels=model.labels_
                k_m=Validation(self.data[:,best_feature==1],km_labels,model.cluster_centers_)
                k_m.evaluation()
                #km_sse.append(k_m.sse())
                for name in all_eva_name:
                    k_m_eva[name].append(k_m.evaluation_list[name])
                
            km_standard = KMeans(n_clusters=self.target_num).fit(self.data[:,best_feature==1])#model.labels_ model.cluster_centers_
            km_std_eva=Validation(self.data[:,best_feature==1],km_standard.labels_,km_standard.cluster_centers_)
            km_std_eva.evaluation()
            km_standard_eva=km_std_eva.evaluation_list[eva_name]
            km_standard_labels=km_standard.labels_
            self.km_standard_cl_size=cluster_size(km_standard_labels)
            self.k_m_best_eva={}
            self.k_m_best_clus_num={}
            self.k_m_best_clus_size={}
            for name in all_eva_name:
                eva_max=max(k_m_eva[name])
                eva_argmax=k_m_eva[name].index(eva_max)
                self.k_m_best_eva[name]=eva_max
                self.k_m_best_clus_num[name]=eva_argmax+2
                self.k_m_best_clus_size[name]=k_m_clus_size[eva_argmax]
            self.k_m_best_clus_labels=k_m_clus_labels[self.k_m_best_clus_num[eva_name]-2]
            self.plot_data(self.data[:,best_feature==1],km_standard_labels,km_standard.cluster_centers_,self.feature_name[best_feature==1],'KM')
            
            #<X-means>
            xm_ave_num=10
            x=self.data[:,best_feature==1]
            xm_ave_clust=[]
            self.x_m_best_eva={}
            for name in all_eva_name:
                self.x_m_best_eva[name]=0.0
            xm_ave_ari=np.zeros(xm_ave_num)
            xm_ave_eva=np.zeros(len(all_eva_name)*xm_ave_num).reshape(len(all_eva_name),xm_ave_num)
            for xm_loop in range(xm_ave_num):
                xm_c = kmeans_plusplus_initializer(data=x, amount_centers=2).initialize()#amount_centers:最初のクラスター数(論文では2を推奨)
                xm_i = xmeans(data=x, initial_centers=xm_c, kmax=cluster_num_max)#kmax:最大クラスタ数,ccore:判定の閾値
                xm_i.process()
                self.x_m_clus_labels=np.zeros(x.shape[0]).astype(dtype=np.int16)
                for i in range(len(xm_i._xmeans__clusters)):
                    buf=np.array(xm_i._xmeans__clusters[i])
                    for j in range(len(buf)):
                        self.x_m_clus_labels[buf[j]]=i
                self.x_m_best_clus_size = cluster_size(self.x_m_clus_labels)
                xm_ave_clust.append(self.x_m_best_clus_size)
                xm_labels=self.x_m_clus_labels
                xm_centers=np.array(xm_i._xmeans__centers)
                x_m=Validation(self.data[:,best_feature==1],xm_labels,xm_centers)
                x_m.evaluation()
                for i,name in enumerate(all_eva_name):
                    xm_ave_eva[i,xm_loop]=x_m.evaluation_list[name]
                xm_ave_ari[xm_loop]=adjusted_rand_score(self.target, self.x_m_clus_labels)
            for i,name in enumerate(all_eva_name):
                self.x_m_best_eva[name]=sum(xm_ave_eva[i])/xm_ave_num
            xm_ari=sum(xm_ave_ari)/xm_ave_num

            self.plot_data(self.data[:,best_feature==1],xm_labels,xm_centers,self.feature_name[best_feature==1],'XM')

            #<調整ランド指数 Adjusted Rand Index：ARI>
            base_ari=adjusted_rand_score(self.target, self.target)
            ecm_ari=adjusted_rand_score(self.target, self.best_labels)
            km_ari=adjusted_rand_score(self.target, km_standard_labels)

            #<正規化相互情報量 Normalized Mutual Information：NMI>
            base_nmi=normalized_mutual_info_score(self.target, self.target)
            ecm_nmi=normalized_mutual_info_score(self.target, self.best_labels)
            km_nmi=normalized_mutual_info_score(self.target, km_standard_labels)
            xm_nmi=normalized_mutual_info_score(self.target, self.x_m_clus_labels)
            
            #シャッフルを元に戻す
            if self.data_cleansing['shuffle']:
                km_standard_labels=DataSelect.reset_shuffle(km_standard_labels)
                self.km_standard_cl_size=cluster_size(km_standard_labels)
                self.k_m_best_clus_labels=DataSelect.reset_shuffle(self.k_m_best_clus_labels)
                self.km_best_clus_size=cluster_size(self.k_m_best_clus_labels)
                self.best_labels=DataSelect.reset_shuffle(self.best_labels)
                self.best_clus_size=cluster_size(self.best_labels)
                self.x_m_clus_labels=DataSelect.reset_shuffle(self.x_m_clus_labels)
                self.x_m_best_clus_size = cluster_size(self.x_m_clus_labels)
            else:self.km_best_clus_size=self.k_m_best_clus_size[eva_name]
            
            #実験結果をファイルに出力（簡易表示）
            file_short=open(self.file_name,'a')
            file_short.write('==========================================================================\n')
            file_short.write("実行日時　　　:{}\n".format(self.run_time))
            file_short.write("完了日時　　　:{}\n".format(now_t))
            file_short.write("データセット　：{}\n".format(self.data_name))
            file_short.write("クラスタ数　　：{}\n".format(self.target_num))
            file_short.write("評価手法　　　：{}\n".format(eva_name))
            file_short.write("---------------------\n")
            file_short.write("{:22s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\n".format(" ","Baseline","ECM","K-means","X-means"))
            file_short.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("Evaluation value",base_eva[eva_name],best_evaluation[eva_name],km_standard_eva,self.x_m_best_eva[eva_name]))#self.k_m_best_eva[eva_name],
            file_short.write("{:22s}\t{:>8d}\t{:>8d}\t{:>8d}\t{:>8d}\n".format("Number of all cluster",self.target_num,len(self.best_clus_size),self.target_num,len(self.x_m_best_clus_size)))#,self.k_m_best_clus_num[eva_name]            
            file_short.write("{:22s}\t{}\t{}\t{}\t{}\n".format("Number of each cluster",self.target_size,self.best_clus_size,self.km_standard_cl_size,self.x_m_best_clus_size))#self.km_best_clus_size
            file_short.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("ARI",base_ari,ecm_ari,km_ari,xm_ari))
            file_short.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("NMI",base_nmi,ecm_nmi,km_nmi,xm_nmi))
            file_short.write('==========================================================================\n')
            file_short.close()

        #実験結果をファイルに出力（詳細表示）
        file=open(self.file_name_all,'a')
        file.write('==========================================================================\n')
        file.write("実行日時　　　:{}\n".format(self.run_time))
        file.write("完了日時　　　:{}\n".format(now_t))
        file.write("データセット　：{}\n".format(self.data_name))
        file.write("クラスタ数　　：{}\n".format(self.target_num))
        file.write("評価手法　　　：{}\n".format(eva_name))
        if sum(best_feature)!=0:
            file.write("---------------------\n")
            file.write("{:22s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\n".format(" ","Baseline","ECM","K-means","X-means"))
            file.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("Evaluation value",base_eva[eva_name],best_evaluation[eva_name],km_standard_eva,self.x_m_best_eva[eva_name]))#self.k_m_best_eva[eva_name],
            file.write("{:22s}\t{:>8d}\t{:>8d}\t{:>8d}\t{:>8d}\n".format("Number of all cluster",self.target_num,len(self.best_clus_size),self.target_num,len(self.x_m_best_clus_size)))#,self.k_m_best_clus_num[eva_name]            
            file.write("{:22s}\t{}\t{}\t{}\t{}\n".format("Number of each cluster",self.target_size,self.best_clus_size,self.km_standard_cl_size,self.x_m_best_clus_size))#self.km_best_clus_size
            file.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("ARI",base_ari,ecm_ari,km_ari,xm_ari))
            file.write("{:22s}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\t{:>8.3f}\n".format("NMI",base_nmi,ecm_nmi,km_nmi,xm_nmi))
            file.write("---------------------\n")
            file.write("<K-means>\n評価値一覧(各クラスタ数の評価値)\n")
            file.write("{}\n".format(k_m_eva))
            file.write("クラスタ数一覧\n")
            file.write("{}\n".format(k_m_clus_size))
            file.write("<X-means>\n評価値一覧\n")
            for name in all_eva_name:file.write("{}:{}\n".format(name,self.x_m_best_eva[name]))
            file.write("評価値平均処理前一覧({}回平均)\n".format(xm_ave_num))
            file.write("Eva:\n{}\n".format(xm_ave_eva))
            file.write("cluster:\n{}\n".format(xm_ave_clust))
            file.write("ARI:\n{}\n".format(xm_ave_ari))
        file.write("----------------\n")
        file.write("データクレンジング：{}\n".format(self.data_cleansing))#-------------------------------------------------------------
        for key in keys:
            if key != 'data':file.write("{:17s} : {}\n".format(key,result[key]))
        file.write("----------------\n")
        file.write("{:17s} : \n{}\n".format('data',result['data']))
        file.write("----------------\n")
        file.write("{}\n".format(keys))
        file.write('==========================================================================\n\n')
        file.close()
        self.run_time=datetime.datetime.today().strftime("%Y/%m/%d-%H:%M:%S")
        
    #評価値更新推移図
    def cycle_fit_fig(self):
        x = np.arange(1,self.cycle_num+1,1)
        plt.plot(list(x),self.cycle_fit,marker='.')
        plt.xlabel('Cycle')
        plt.ylabel('Fitness')
        img_dir=self.directory_name+'\\\\画像'
        os.makedirs(img_dir, exist_ok=True)#ディレクトリの新規作成
        fig_name=img_dir+'\\\\'+self.file_name_base+datetime.datetime.today().strftime("_%H-%M-%S")+'_'+self.eva_name+'.png'
        plt.savefig(fig_name)
        plt.close()

    #クラスタの状態を出力
    def plot_data(self,data,labels,centers,feature_name,name=''):
        if data.shape[1]==2:#特徴選択によって，特徴量が2のとき結果を出力
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            cl_size = cluster_size(labels)

            for i in range(len(cl_size)):
                ax.scatter(data[:, 0][labels==i], data[:, 1][labels==i], color=colors[i], alpha=0.5,label='Cluster {}'.format(i+1))
                ax.scatter(centers[i, 0], centers[i, 1], marker='*', color=colors[i],edgecolor='w', s=300,label='Center of cluster {}'.format(i+1))

            plt.legend(loc='upper left')
            plt.xlabel(feature_name[0])
            plt.ylabel(feature_name[1])
            img_dir=self.directory_name+'\\\\画像'
            os.makedirs(img_dir, exist_ok=True)#ディレクトリの新規作成
            fig_name=img_dir+'\\\\'+self.file_name_base+datetime.datetime.today().strftime("_%H-%M-%S")+'_'+self.eva_name+'_'+name+'cluster.png'
            plt.savefig(fig_name)
            plt.close()



#データ出力用ファイル名指定
ProgName='ECM+ABC'

#使用するデータセットを指定
dataset_list="seeds user-knowledge-modeling iris wine glass thyroid raisin breast-cancer breast-cancer-wisconsin".split()

#使用する評価手法を指定
evaluation_list="ch dunn pbm sil db xb".split()

setup()
for use_data in dataset_list:
    print(use_data)
    test_data=DataSelect(use_data)
    test_save=OutputResult(ProgName,test_data.data_name,test_data.data_cleansing,test_data.target,test_data.shuffle_sele,test_data.feature_name)
    for eva_name in evaluation_list: 
        print(eva_name)

        #提案手法の条件設定
        test_ecm_abc=ABC(eva_name,test_data.data,test_data.target,feature_sele_dthr=0.0)# ECM+ABC(特徴選択なし)
        #test_ecm_abc=ABC(eva_name,test_data.data,test_data.target)#                      ECM+ABC(特徴選択あり)
        #test_ecm_abc=ABC(eva_name,test_data.data,test_data.target,mabc_run=True)#        ECM+MABC
        
        test_ecm_abc.run(1000)#繰り返し回数（ABCアルゴリズムの繰り返し回数を指定）
        test_save.save(test_ecm_abc.__dict__)
print("------------------End------------------")