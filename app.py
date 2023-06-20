import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score


def load_autoencoder_model():
    autoencoder = tf.keras.models.load_model('D:/WebDevelopment/PlayCluster/autoencoder.h5')
    return autoencoder


def main():
    st.title("PlayCluster")

    file_path1 = st.file_uploader("Upload the first CSV file:", type="csv")
    file_path2 = st.file_uploader("Upload the second CSV file:", type="csv")

    if file_path1 and file_path2:
        p1 = pd.read_csv(file_path1)
        p2 = pd.read_csv(file_path2)

        autoencoder = load_autoencoder_model()
        kpca=KernelPCA(n_components=2, kernel="linear")
        p1_autoencoded = autoencoder.predict([pd.get_dummies(p1.select_dtypes(include = ['object'])), p1.select_dtypes(include =np.number)])
        p1_kpca=kpca.fit_transform(p1_autoencoded)
        max_sil=0
        for i in range(2,10):
            model=AgglomerativeClustering(n_clusters = i, linkage="ward")
            agg_pred_p1=model.fit_predict(p1_kpca)
            silhouette_avg = silhouette_score(p1_kpca, agg_pred_p1)
            if(silhouette_avg > max_sil):
                    max_sil = silhouette_avg
                    p1_clusters = i
                    max_pred_p1 = agg_pred_p1
                    final_df_p1 = p1_kpca
        p1_sil = silhouette_score(final_df_p1, max_pred_p1)
        p1_db = davies_bouldin_score(final_df_p1, max_pred_p1)

        p2_autoencoded = autoencoder.predict([pd.get_dummies(p2.select_dtypes(include = ['object'])), p2.select_dtypes(include=np.number)])
        p2_kpca=kpca.fit_transform(p2_autoencoded)
        max_sil=0
        for i in range(2,10):
            model=AgglomerativeClustering(n_clusters = i, linkage="ward")
            agg_pred_p2=model.fit_predict(p2_kpca)
            silhouette_avg = silhouette_score(p2_kpca, agg_pred_p2)
            if(silhouette_avg > max_sil):
                    max_sil = silhouette_avg
                    p2_clusters = i
                    max_pred_p2 = agg_pred_p2
                    final_df_p2 = p2_kpca
        p2_sil = silhouette_score(final_df_p2, max_pred_p2)
        p2_db = davies_bouldin_score(final_df_p2, max_pred_p2)


        p12 = pd.concat([p1, p2], axis = 0)
        p12_autoencoded = autoencoder.predict([pd.get_dummies(p12.select_dtypes(include = ['object'])), p12.select_dtypes(include=np.number)])
        p12_kpca=kpca.fit_transform(p12_autoencoded)
        max_sil=0
        for i in range(2,10):
            model=AgglomerativeClustering(n_clusters = i, linkage="ward")
            agg_pred_p12=model.fit_predict(p12_kpca)
            silhouette_avg = silhouette_score(p12_kpca, agg_pred_p12)
            if(silhouette_avg > max_sil):
                    max_sil = silhouette_avg
                    p12_clusters = i
                    max_pred_p12 = agg_pred_p12
                    final_df_p12 = p12_kpca
        p12_sil = silhouette_score(final_df_p12, max_pred_p12)
        p12_db = davies_bouldin_score(final_df_p12, max_pred_p12)
        
        # SI_sil = ((p12_sil*2/(p1_sil+p2_sil)) * ((p12_clusters/(min(p1_clusters, p2_clusters)))))
        # SI_db = ((p1_db+p2_db/(p12_db*2)) * ((p12_clusters/(min(p1_clusters, p2_clusters)))))

        SI_new = (abs(p1_db - p2_db)/abs(p1_sil-p2_sil)) * ((p12_clusters/(min(p1_clusters, p2_clusters))))
        #SI_new = 1/(1 + math.pow(math.e, -(SI_new)))
        st.title('Silouette score:')
        st.write('Playlist 1: ',p1_sil)
        st.write('Playlist 2: ' ,p2_sil)
        st.write('Combined: ' ,p12_sil)

        st.title('DB Score:')
        st.write('Playlist 1: ',p1_db)
        st.write('Playlist 2: ' ,p2_db)
        st.write('Combined: ' ,p12_db)

        st.title('No.of clusters')
        st.write('Playlist 1: ',p1_clusters)
        st.write('Playlist 2: ' ,p2_clusters)
        st.write('Combined: ' ,p12_clusters)

        #st.write(SI_sil, ' ', SI_db)
        st.title('Similarity Index')
        st.write(SI_new)
        
if __name__ == '__main__':
    main()
