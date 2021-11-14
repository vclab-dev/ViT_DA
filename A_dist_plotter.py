import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
sns.set_style("darkgrid")

def a_dist(acc):
    "Given the accuracy get the A-Distance"
    return abs(2*(0.02*acc - 1))

df = pd.read_csv('a_dist.csv')
df['A-Distance'] = df['accuracy'].apply(a_dist)
df = df.drop([12,13, 18, 19])
print(df)

plt.ylim(1, 2)
sns.barplot(x="Adaptation", y="A-Distance", hue="model", data=df)

plt.savefig('a_dist.pdf')