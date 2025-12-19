import numpy as np
import pandas as pd
from scipy import stats

# ====================================================================
# DADOS EXTRAÍDOS DAS TABELAS DO SEU PDF
# ====================================================================

# Cenários de 10 minutos (Tabela 3)
data_10min = {
    'Cenario': ['Quarta-Jantar', 'Segunda-Almoço', 'Segunda-Jantar', 'Quarta-Almoço'],
    'GT': [8, 48, 20, 23],
    'HOG_Detect': [3, 1, 2, 1],
    'YOLO_Detect': [7, 69, 23, 28],
    'MediaPipe_Detect': [4, 0, 3, 3]
}

# Cenários de 30 minutos (Tabela 4)
data_30min = {
    'Cenario': ['Quarta-Jantar', 'Segunda-Almoço', 'Segunda-Jantar', 'Quarta-Almoço'],
    'GT': [36, 92, 43, 65],
    'HOG_Detect': [9, 0, 8, 4],
    'YOLO_Detect': [40, 85, 66, 93],
    'MediaPipe_Detect': [13, 15, 10, 18]
}

# Combinar todos os 8 cenários
all_gt = np.array(data_10min['GT'] + data_30min['GT'])
all_hog = np.array(data_10min['HOG_Detect'] + data_30min['HOG_Detect'])
all_yolo = np.array(data_10min['YOLO_Detect'] + data_30min['YOLO_Detect'])
all_mp = np.array(data_10min['MediaPipe_Detect'] + data_30min['MediaPipe_Detect'])

print("="*80)
print("CÁLCULO DAS MÉTRICAS ESTATÍSTICAS REAIS")
print("="*80)
print(f"\nDados utilizados: {len(all_gt)} cenários (4 de 10min + 4 de 30min)")
print(f"Ground Truth: {all_gt}")
print(f"HOG+SVM:      {all_hog}")
print(f"YOLOv8n:      {all_yolo}")
print(f"MediaPipe:    {all_mp}")

# ====================================================================
# FUNÇÕES DE CÁLCULO
# ====================================================================

def calcular_metricas(y_true, y_pred, nome):
    """Calcula todas as métricas estatísticas"""
    
    # MAE - Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE - Root Mean Square Error
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    # MAPE - Mean Absolute Percentage Error (evitando divisão por zero)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    
    # Correlação de Pearson
    if np.std(y_pred) > 0:  # Evitar erro se todas previsões forem iguais
        correlacao, p_valor = stats.pearsonr(y_true, y_pred)
    else:
        correlacao, p_valor = 0.0, 1.0
    
    # Viés (Bias) - média dos erros (positivo = superestimação)
    vies = np.mean(y_pred - y_true)
    
    # Desvio padrão dos erros
    erro_std = np.std(y_pred - y_true)
    
    # Percentual de acertos dentro de ±10% do valor real
    tolerancia_10pct = np.sum(np.abs((y_pred - y_true) / np.maximum(y_true, 1)) <= 0.10) / len(y_true) * 100
    
    return {
        'Algoritmo': nome,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Correlacao': correlacao,
        'P_valor': p_valor,
        'Vies': vies,
        'Erro_STD': erro_std,
        'Acerto_10pct': tolerancia_10pct
    }

# ====================================================================
# CÁLCULO PARA CADA ALGORITMO
# ====================================================================

resultados = []

print("\n" + "="*80)
print("RESULTADOS DETALHADOS POR ALGORITMO")
print("="*80)

for nome, y_pred in [('HOG + SVM', all_hog), 
                     ('YOLOv8n', all_yolo), 
                     ('MediaPipe Pose', all_mp)]:
    
    metricas = calcular_metricas(all_gt, y_pred, nome)
    resultados.append(metricas)
    
    print(f"\n{nome}:")
    print(f"  MAE (Erro Médio Absoluto):           {metricas['MAE']:.2f} pessoas")
    print(f"  RMSE (Raiz Erro Quadrático Médio):   {metricas['RMSE']:.2f} pessoas")
    print(f"  MAPE (Erro Percentual Médio):        {metricas['MAPE']:.1f}%")
    print(f"  Correlação de Pearson (r):           {metricas['Correlacao']:.3f}")
    print(f"  P-valor:                              {metricas['P_valor']:.4f}")
    print(f"  Viés médio:                           {metricas['Vies']:+.2f} pessoas")
    print(f"  Desvio padrão dos erros:              {metricas['Erro_STD']:.2f} pessoas")
    print(f"  Acertos dentro de ±10%:               {metricas['Acerto_10pct']:.0f}%")

# ====================================================================
# TABELA RESUMO
# ====================================================================

df_resultados = pd.DataFrame(resultados)

print("\n" + "="*80)
print("TABELA RESUMO")
print("="*80)
print(df_resultados[['Algoritmo', 'MAE', 'RMSE', 'MAPE', 'Correlacao']].to_string(index=False))

# ====================================================================
# CÓDIGO LATEX PRONTO PARA COPIAR
# ====================================================================

print("\n" + "="*80)
print("CÓDIGO LATEX (copie e cole no seu documento):")
print("="*80)

print("""
\\begin{table}[!ht]
\\centering
\\caption{\\label{tabela:metricas_erro}Métricas de erro dos algoritmos em todos os cenários}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Algoritmo} & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{MAPE (\\%)} & \\textbf{Correlação (r)} \\\\
\\midrule""")

for _, row in df_resultados.iterrows():
    print(f"{row['Algoritmo']} & {row['MAE']:.1f} & {row['RMSE']:.1f} & {row['MAPE']:.1f} & {row['Correlacao']:.2f} \\\\")

print("""\\bottomrule
\\end{tabular}
\\caption*{\\footnotesize MAE = Erro Médio Absoluto; RMSE = Raiz do Erro Quadrático Médio; MAPE = Erro Percentual Médio Absoluto. Fonte: Dados experimentais da autora, 2025.}
\\end{table}
""")

# ====================================================================
# INTERPRETAÇÃO DOS RESULTADOS
# ====================================================================

print("\n" + "="*80)
print("INTERPRETAÇÃO PARA O TEXTO:")
print("="*80)

melhor_mae = df_resultados.loc[df_resultados['MAE'].idxmin()]
melhor_rmse = df_resultados.loc[df_resultados['RMSE'].idxmin()]
melhor_corr = df_resultados.loc[df_resultados['Correlacao'].idxmax()]

print(f"\n✓ Melhor MAE: {melhor_mae['Algoritmo']} ({melhor_mae['MAE']:.1f} pessoas)")
print(f"✓ Melhor RMSE: {melhor_rmse['Algoritmo']} ({melhor_rmse['RMSE']:.1f} pessoas)")
print(f"✓ Melhor Correlação: {melhor_corr['Algoritmo']} (r = {melhor_corr['Correlacao']:.2f})")

print("\nInterpretação da Correlação:")
for _, row in df_resultados.iterrows():
    r = row['Correlacao']
    if r >= 0.9:
        interpretacao = "muito forte"
    elif r >= 0.7:
        interpretacao = "forte"
    elif r >= 0.5:
        interpretacao = "moderada"
    elif r >= 0.3:
        interpretacao = "fraca"
    else:
        interpretacao = "muito fraca"
    print(f"  {row['Algoritmo']}: r = {r:.2f} → correlação {interpretacao}")

print("\nViés (tendência de super/subestimação):")
for _, row in df_resultados.iterrows():
    if row['Vies'] > 0:
        tendencia = f"superestima em média {abs(row['Vies']):.1f} pessoas"
    elif row['Vies'] < 0:
        tendencia = f"subestima em média {abs(row['Vies']):.1f} pessoas"
    else:
        tendencia = "sem viés aparente"
    print(f"  {row['Algoritmo']}: {tendencia}")

print("\n" + "="*80)
print("Pronto! Use os valores acima no seu texto.")
print("="*80)