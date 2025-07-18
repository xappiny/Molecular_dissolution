import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('\features.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
ax = sns.histplot(df['Drug_dose'].dropna(), bins=10, kde=True, color='teal')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)  
ax.spines['bottom'].set_linewidth(1.5)  

plt.grid(linestyle='', alpha=0.7) 
plt.tick_params(axis='both', labelsize=16)
plt.xlabel('Drug dose (mg)', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Distribution of Drug dose', fontsize=20)

plt.savefig("\Drug_dose.png", dpi=600, bbox_inches='tight')

plt.show()
