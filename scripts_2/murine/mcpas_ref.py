import pandas as pd
import numpy as np

df = pd.read_csv('../../Data/McPAS-TCR.csv')
df = df[df['Species']=='Mouse']