import matplotlib.pyplot as plt
import numpy as np

def a_dist(acc):
    "Given the accuracy get the A-Distance"
    return 2*(0.02*acc - 1)

  
# create data
# order is AD, AW, WD
x = np.arange(3)
Res50 = [99.9,99.7,98.89] # Domain accuracy for resnet
Res50 = [a_dist(acc) for acc in Res50] # A-distance for Resnet
ViT = [97.22,90.46,90.12] # Domain Accuracy for ViT
ViT = [a_dist(acc) for acc in ViT] # A-distance for ViT
width = 0.2
  
# plot data in grouped manner of bar type
fig = plt.figure(figsize=(12,8))
plt.grid(True,axis='y',color='black', linestyle='--', linewidth=.2)
plt.bar(x-0.2, Res50, width, color='orange')
plt.bar(x, ViT, width, color='green')
plt.xticks(x-0.1, ['A&D', 'A&W', 'W&D'])
plt.title("Office-31 A-Distance(Res50 Vs Vit)")
plt.xlabel("Source-Target pair")
plt.ylabel("A-Distance")
plt.legend(["Res50", "ViT"],loc="best")
fig.savefig("A_Distnace.png",dpi=150)
plt.show()
