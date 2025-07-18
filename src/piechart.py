import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('\AUC_features.csv')

df['Diss_pH_grouped'] = df['Diss_pH'].apply(lambda x: 'pH ≤ 2' if x in [1.2, 1.6, 2] else f'pH:{x}')

process_counts = df['Diss_pH_grouped'].value_counts()
print(process_counts)

colors = ['#70AED4', '#B0D1E4', '#FFF2CD', '#FFF8E5', '#D3E2B7', '#90A8B6', '#E6C9C9', '#D4B5B0']

plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0.05, right=0.65, top=0.8, bottom=0.2)

def my_autopct(pct):
    return f'{pct:.1f}%' if pct > 5 else ''

wedges, texts, autotexts = plt.pie(
    process_counts,
    colors=colors[:len(process_counts)],
    autopct=my_autopct,
    wedgeprops={'edgecolor': 'white', 'width': 0.5},
    pctdistance=0.8,
    textprops={'fontsize': 16}
)

for autotext in autotexts:
    autotext.set_fontsize(20)

plt.title('Dissolution pH', fontsize=20, pad=20)

plt.legend(
    process_counts.index,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    frameon=False,
    fontsize=14
)

plt.axis('equal')

plt.savefig(
    r'\Diss_pH_pie_chart_grouped.png',
    dpi=600,
    bbox_inches='tight',
    transparent=True
)
plt.show()
