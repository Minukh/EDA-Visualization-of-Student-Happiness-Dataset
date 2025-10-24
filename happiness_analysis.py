import numpy as np
import pandas as pd
import seaborn as sns
import  matplotlib.pyplot as plt

df = pd.read_csv('happiness_train_dataset.csv')
numeric_df = df.select_dtypes(include=[np.number])
print(df.describe())

sns.histplot(df['Happiness_Level'], kde=True, color='red')
plt.title('Distribution of Happiness Levels')
plt.xlabel('Happiness Level')
plt.ylabel('Number of students')
plt.show()

sns.scatterplot(x=df['Academic_Stress'], y=df['Happiness_Level'])
plt.title("Academic Stress vs. Happiness Level")
plt.xlabel('Academic Stress')
plt.ylabel('Happiness Level')
plt.show()

sns.heatmap(numeric_df.corr(), cmap='coolwarm',annot=True)
plt.title("Happiness Level vs. Happiness Level")
plt.show()