import os
import pandas as pd


archivos_tsv = [archivo for archivo in os.listdir('.') if archivo.endswith('.tsv')]


df_total = pd.DataFrame()

for archivo in archivos_tsv:
 
    df = pd.read_csv(os.path.join('.', archivo), sep='\t')
    df_total = pd.concat([df_total, df])


df_total.to_csv('datos_combinados.tsv', sep='\t', index=False)
