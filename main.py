from tkinter import ttk
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

#Leer con pandas el dataset Iris
dataset = pd.read_csv('Iris.csv')
print(dataset)
x = dataset.iloc[:,:4].values
y = dataset['species'].values
#dataset por cada especie

ventana = tk.Tk()
ventana.title("K.means")
ventana.geometry("720x550")
ventana.configure(bg="azure")


titulo = tk.Label(ventana, text="K-means", bg = "azure", font=("Helvetica", 20))
titulo.pack(fill=tk.X)
ltit = tk.Label(ventana, text="Matriz de confusión", bg="azure", font=("Helvetica", 16))
ltit.place(x=20, y=40)
lbl = tk.Label(ventana, text="set|ver|vir", bg="azure", font=("Helvetica", 12))
lbl.place(x=80, y=70)
lbl2 = tk.Label(ventana, text="setosa", bg="azure", font=("Helvetica", 12))
lbl2.place(x=10, y=90)
lbl3 = tk.Label(ventana, text="versicolor", bg="azure", font=("Helvetica", 12))
lbl3.place(x=10, y=110)
lbl4 = tk.Label(ventana, text="virginica", bg="azure", font=("Helvetica", 12))
lbl4.place(x=10, y=130)



lbl6=tk.Label(ventana, text="selecciona el numero de clusters:",bg="azure", font=("Helvetica", 14))
lbl6.place(x=250,y=50)
combo_cluster=ttk.Combobox(ventana, values=["2","3","4","5","6"])
combo_cluster.set(3)
combo_cluster.place(x=250, y=100)


ltitd2 = tk.Label(ventana, text="Dataset", bg="azure", font=("Helvetica", 16))
ltitd2.place(x=200, y=200)
caja2 = tk.Text(ventana)
caja2.insert(tk.INSERT,dataset)
caja2.place(x=10, y=240, height=300, width=700)
def graf():
    # Visualising the clusters
    nu = int(combo_cluster.get())
    print(nu)
    kmeans = KMeans(n_clusters=nu, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)
    d = {"setosa": 0, "versicolor": 1, "virginica": 2}
    mc = confusion_matrix(dataset.species.map(d), y_kmeans)
    lbl5 = tk.Label(ventana, text=mc, bg="azure", font=("Helvetica", 12))
    lbl5.place(x=80, y=92)
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Virginica')
    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='yellow', label='Centroides')
    plt.legend()
    plt.show()
boton_1 = tk.Button(ventana, text="Mostrar", width=8, command=graf)
boton_1.place(x=450, y=95)
ventana.mainloop()