# x-vector using pytorch
owner: https://github.com/KrishnaDN/x-vector-pytorch#installation

# Beüzelemlés
1. Felkell telepíteni a requirements.txtből mindent
2. +conda fontos
3. https://pytorch.org/get-started/locally/
 - Itt windows conda python CUDA 10.2
 
# Mi van kész
1. X-vector kész
2. Feature extractor kész
3. Downsampling nagyjából kész az alapján kellene feature extractolni

# Ami hátra van
1. SVM megnézése betanitása
2. # FONTOS LENNE FELRAKNI A META MAPPÁT VALAHOVA
https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python#building



5 cím
-> x-vector
-> svm
-> downsampling aibora
uar

svm  -> standardscaler -> 
svm fit csak tanitó többi 
10^-5 10 complexity
svm -> posterior/prob
kernel: rbf -> paramétere: complexity + gamma

amelyik deven a legjobb -> teszt

Eddig van betanított x-vector 90% körüli a sima bea-son
TEHÁT megvannak a feature extractok minden hangra külön label mappákba.
Pontosan ez a downsampling mire jó?
SVM-nél standardscaler kell és csak tanitóra kell a fit.
Többire mit pontosan?
Hogy kellene beadni neki a labeleket kezdek elveszni.
Hogy kellene ezt az svm-et tanitani pontosan.

https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/
https://torchbearer.readthedocs.io/en/0.1.7/examples/svm_linear.html

#Steps
1. datasets.py-al lefuttatni a-t, amibe csak az összes hangfájl útját kell megadni
2. Training training_xvector.py átkell állítani amennyi class van (len(a.py által létrehozott mappák))
3. A legjobb model alapján feature_extractor.py 
4. SVM.py lefuttatni és megvannak az értékek