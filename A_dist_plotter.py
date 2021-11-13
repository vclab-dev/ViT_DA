import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
sns.set_style("darkgrid")

df = pd.read_csv('a_dist.csv')
plt.ylim(22, 37)
sns.barplot(x="adaptation", y="accuracy", hue="model", data=df)

plt.savefig('output.png')