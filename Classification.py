import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from skimage.measure import label, regionprops
from skimage import img_as_float, img_as_ubyte
from skimage import io

import pandas as pd
from skimage import morphology
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mayavi import mlab
from math import sqrt,pi
from numpy import random
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import calinski_harabasz_score
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors as NN
from sklearn import decomposition






stack_30=io.imread('C:/Users/mjkallungal/Desktop/Jesbeer/PhD/Experiments/Charac/tomo/09-30/ep50CB_30tr/nlm_rw_30_edges removed.tif')
stack_60=io.imread('C:/Users/mjkallungal/Desktop/Jesbeer/PhD/Experiments/Charac/tomo/09-30/ep50CB_60tr/nlm_rw_60_latest_edges removed.tif')
stack_30_in=io.imread('C:/Users/mjkallungal/Desktop/Jesbeer/PhD/Experiments/Charac/tomo/12-05/f1/f1_rw_30_edges removed.tif')
stack_60_in=io.imread('C:/Users/mjkallungal/Desktop/Jesbeer/PhD/Experiments/Charac/tomo/12-05/f2/f2_rw_60_edges removed.tif')
df_30=features(stack_30,1)
df_60=features(stack_60,1)
df_30_in=features(stack_30_in,1)
df_60_in=features(stack_60_in,1)
cal_tot=[]
db_tot=[]
sil_tot=[]
for i in range(10):
    from sklearn.preprocessing import StandardScaler
    X=df_30.iloc[:,0:1].join(df_30.iloc[:,2:4])
    X_30=X.copy()
    X_60=df_60.iloc[:,0:1].join(df_60.iloc[:,2:4])
    cross=X_30.append(X_60)
    cross=cross.append(df_30_in.iloc[:,0:1].join(df_30_in.iloc[:,2:4]))
    cross=cross.append(df_60_in.iloc[:,0:1].join(df_60_in.iloc[:,2:4]))
    X_cross=np.array(cross.sample(frac=0.3, replace=True, random_state=np.random.randint(100)),dtype=float)
    scale=StandardScaler()
    scaler= scale.fit(X_cross)
    X_train=scaler.transform(X_cross)
    X_test_30=scaler.transform(X)
    X_test_60=scaler.transform(X_60)
    X_30_test_in=scaler.transform(df_30_in.iloc[:,0:1].join(df_30_in.iloc[:,2:4]))
    X_60_test_in=scaler.transform(df_60_in.iloc[:,0:1].join(df_60_in.iloc[:,2:4]))
    Sum_of_squared_distances = []
    K = range(2,15)
    cal=[]
    db=[]
    sil=[]
    for k in K:
        km = KMeans(n_clusters=k,init='random', n_init=20,max_iter=500,random_state=12,n_jobs=-1,algorithm='elkan')
        kmeans = km.fit(pd.DataFrame(X_train))
        Sum_of_squared_distances.append(km.inertia_)
        labels = kmeans.labels_
        sil.append(silhouette_score(X_train, labels))
        #cal.append(calinski_harabasz_score(X_train, labels))
        #db.append(davies_bouldin_score(X_train, labels))
        #print(" DBI {} :".format(k),davies_bouldin_score(X_train, labels))
        #print(" CAH {} :".format(k),cal[k-2])
    #cal_tot.append(cal)
    #db_tot.append(db)
    sil_tot.append(sil)

for i in range(len(cal_tot)):
    #plt.plot(range(2,15),cal_tot[i])
    
plt.show()

# lab=list(map(float, input("resolution: ").split()))
#res = list(map(int, input("no of clusters: ").split()))
km = KMeans(n_clusters=9,init='random', n_init=20,max_iter=1000,random_state=0,n_jobs=-1,algorithm='elkan')
kmeans = km.fit(X_train)
True_sample=pd.DataFrame(X_cross)
x=kmeans.cluster_centers_

def add_centroid(df,X_test_30,True_sample,x,stack,label,res):
    kmeans_pred = kmeans.predict(X_test_30)
    #kmeans_pred_60 = km.predict(X_test_60)
    df['label']= kmeans_pred
    #df_60['label']= kmeans_pred_60 
    z=[]
    for i in range(len(x)):
        z.append(True_sample.std()*x[i]+True_sample.mean())
        df.loc[df.label==i,'c_V']=z[i][0]
        #df.loc[df.label==i,'c_eq_diam']=z[i][1]
        df.loc[df.label==i,'c_B']=z[i][1]
        df.loc[df.label==i,'c_E^2']=z[i][2]
        #df.loc[df.label==i,'c_theta']=z[i][4]
    df['Vol_fraction']=df.V*100/(stack.shape[0]*stack.shape[1]*stack.shape[2]*res**3)
    df['mix']=label 
    return df

labe=['30_rpm','60_rpm','30_rpm_td','60_rpm_tf']

df_30=add_centroid(df_30,X_test_30,True_sample,x,stack_30,labe[0],1)
df_60=add_centroid(df_60,X_test_60,True_sample,x,stack_60,labe[1],1)
df_30_in=add_centroid(df_30_in,X_30_test_in,True_sample,x,stack_30_in,labe[2],1)
df_60_in=add_centroid(df_60_in,X_60_test_in,True_sample,x,stack_60_in,labe[3],1)

def features(image,res):
    clean_60 = morphology.remove_small_objects(image == 255, 125)
    stack_label_60 = label(clean_60,neighbors=8,background=0,connectivity=1)
    regions_60=regionprops(stack_label_60)
    label_number=np.arange(np.max(stack_label_60)+1)
    print(label_number)

    volume=[]
    density=[]
    eccentricity=[]
    eq_diam=[]
    maj_length=[]
    min_length=[]
    bulk=[]
    centroid=[]
    area=[]
    bobox=[]
    theta=[]
    theta_x=[]
    theta_y=[]
    tensor=[]
    feret_diameter=[]
    cords=[]
    hull_c=[]
    for props in regions_60:
        volume.append(props.area)
        eq_diam.append(props.equivalent_diameter)
        bulk.append(props.extent)
        maj_length.append(props.major_axis_length)
        min_length.append(props.minor_axis_length)
        centroid.append(props.centroid)
        bobox.append(props.bbox)
        tensor.append(props.inertia_tensor)
        hull_c.append(props.convex_area)
    
    
    for i in range(len(tensor)):
        evals,evecs=np.linalg.eig(tensor[i])
        sort_indices = np.argsort(evals)[::1]
        z_v1,y_v1, x_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
#       theta.append(np.arccos(abs(z_v1)/(sqrt(z_v1**2+y_v1**2+x_v1**2))))
        theta_x.append(np.arccos(abs(x_v1)/(sqrt(z_v1**2+y_v1**2+x_v1**2))))
        theta_y.append(np.arccos(abs(y_v1)/(sqrt(z_v1**2+y_v1**2))))
        theta.append(np.arccos(abs(z_v1)/(sqrt(z_v1**2+y_v1**2))))
    tables = pd.DataFrame(np.column_stack((volume,
                                           eq_diam,
                                           hull_c,
                                           eq_diam,
                                           min_length,
                                           maj_length,
                                           eq_diam,
                                           theta,
                                           theta_x,
                                           theta_y,
                                           centroid,
                                           bobox)))
    df_60=tables.rename(columns={0:"V",
                                 1:"Eq_diam",
                                 2:"B",
                                 3:"E^2",
                                 4:"b",
                                 5:"a",
                                 6:"Q",
                                 7:"Theta",
                                 8:"Theta_x",
                                 9:"Theta_y",
                                 10:"centroid_z",
                                 11:"centroid_y",
                                 12:"centroid_x"})

    
    df_60['B']=1-df_60.V/df_60.B
    df_60['Q']=df_60.b/df_60.a
    df_60['E^2']=(1-(df_60.b/df_60.a)**2)**0.5
    df_60['V']=df_60['V']*res**3
    df_60['Eq_diam']=df_60['Eq_diam']*res
    df_60['b']=df_60['b']*res
    df_60['a']=df_60['a']*res
#    df_60['T']=(pi*df_60.a**3/3-df_60.V)/df_60.V
    
    return df_60





