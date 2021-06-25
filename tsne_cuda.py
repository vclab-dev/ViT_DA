import numpy as np
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
import os, shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 1000
sns.set(style="darkgrid")

save_dir = 'tsne_plots_cuda'
store = np.load('saved_features/amazon_features.npy', allow_pickle=True)

###             [[feat,cls], [feat,cls] ...]

shutil.rmtree(save_dir)
os.mkdir(save_dir)

X = np.array(store[:,0].tolist())
label = store[:,1] 

data = {}


lr = [5,7,10,13,15,18,20,25,30]
for i in range(5,25):
    f = plt.figure(figsize=(50,50))
    for count,j in enumerate(lr):
        print(f'TSNE Running for perplexity={i} and lr={j}')
        X_embedded = TSNE(perplexity=i, learning_rate=j).fit_transform(X)
        x,y = X_embedded[:,0],X_embedded[:,1]
        data['x'], data['y'], data['label'] = x,y, label
        df=pd.DataFrame(data)
        ax = f.add_subplot(round(len(lr)**(0.5)),round(len(lr)**(0.5)),count+1)
        sns.scatterplot(data=df, x="x", y="y", hue="label", legend=False)

        # break
    plt.savefig(f'{save_dir}/tsne_per_{i}.png')
    plt.clf()
    print('Saved! ' + f'{save_dir}/tsne_per_{i}.png')
    # break
print('Completed all')
