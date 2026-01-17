import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os

def analyze_detector_log(csv_file):
    print(f"--- Processando arquivo: {csv_file} ---")
    
    # 1. Carregar Dados
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Erro crítico ao ler o CSV: {e}")
        return

    # 2. Cálculos Estatísticos Básicos
    total_samples = len(df)
    if total_samples == 0:
        print("O arquivo está vazio.")
        return

    start_time = df['elapsed_time'].min()
    end_time = df['elapsed_time'].max()
    duration = end_time - start_time
    
    class_counts = df['class_name'].value_counts()
    alert_counts = df['alert_status'].value_counts()
    
    avg_confidence = df.groupby('class_name')['confidence'].mean()
    
    # 3. Análise de Alertas
    real_alerts = df[df['alert_status'] == 'ALERTA_JAMMER'].shape[0]
    ignored_alerts = df[df['alert_status'] == 'IGNORED_LOW_CONF'].shape[0]
    cooldown_events = df[df['alert_status'] == 'COOLDOWN'].shape[0]
    
    # 4. Geração dos Gráficos
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Relatório de Detecção Passiva de Jammers\nArquivo: {csv_file}', fontsize=16)
    
    # --- GRÁFICO 1: Timeline ---
    ax1 = plt.subplot(3, 1, 1)
    
    colors = {'Normal (Ocioso)': 'green', 'Normal (Dados)': 'blue', 'JAMMER': 'red'}
    row_colors = df['class_name'].map(colors).fillna('gray')
    
    ax1.scatter(df['elapsed_time'], df['confidence'], c=row_colors, s=15, alpha=0.6, label='Classificação')
    
    # Marca os alertas
    alerts = df[df['alert_status'] == 'ALERTA_JAMMER']
    for _, row in alerts.iterrows():
        ax1.axvline(x=row['elapsed_time'], color='red', linestyle='--', alpha=0.8)
        # ax1.text(row['elapsed_time'], 105, 'ALERTA', color='red', fontsize=8, rotation=90) # Texto opcional se ficar muito cheio

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Ocioso', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Dados', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Jammer (Detectado)', markerfacecolor='red', markersize=8),
        Line2D([0], [0], color='red', linestyle='--', label='Alerta Confirmado')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    ax1.set_ylabel('Confiança (%)')
    ax1.set_title('Timeline: Confiança de Detecção e Alertas')
    ax1.set_ylim(0, 115)
    ax1.grid(True)

    # --- GRÁFICO 2: Pizza (Distribuição) ---
    ax2 = plt.subplot(3, 2, 3)
    if not class_counts.empty:
        pie_colors = [colors.get(x, 'gray') for x in class_counts.index]
        ax2.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=140)
        ax2.set_title('Distribuição de Classes (Amostras)')
    else:
        ax2.text(0.5, 0.5, 'Sem dados', ha='center')

    # --- GRÁFICO 3: Barras (Status de Alerta) ---
    ax3 = plt.subplot(3, 2, 4)
    if not alert_counts.empty:
        # Remove 'MONITORANDO' para focar nos eventos interessantes
        counts_to_plot = alert_counts.drop('MONITORANDO', errors='ignore')
        if not counts_to_plot.empty:
            counts_to_plot.plot(kind='bar', ax=ax3, color='purple', alpha=0.7)
            ax3.set_title('Eventos do Sistema (Excluindo Monitoramento)')
            ax3.set_ylabel('Contagem')
            ax3.tick_params(axis='x', rotation=0)
            for i, v in enumerate(counts_to_plot):
                ax3.text(i, v + 0.1, str(v), ha='center')
        else:
            ax3.text(0.5, 0.5, 'Apenas monitoramento normal', ha='center')

    # --- GRÁFICO 4: Boxplot (Confiança) ---
    ax4 = plt.subplot(3, 1, 3)
    data_to_plot = [df[df['class_name'] == cls]['confidence'] for cls in df['class_name'].unique()]
    labels = df['class_name'].unique()
    
    if len(data_to_plot) > 0:
        ax4.boxplot(data_to_plot, labels=labels, vert=False, patch_artist=True)
        ax4.set_title('Distribuição de Confiança por Classe')
        ax4.set_xlabel('Confiança (%)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 5. Salvar e Relatar
    output_img = csv_file.replace('.csv', '_analise.png')
    plt.savefig(output_img)
    print(f"Gráfico salvo em: {output_img}")
    
    print("\n" + "="*50)
    print("RELATÓRIO DE DETECÇÃO DE JAMMERS")
    print("="*50)
    print(f"Arquivo: {csv_file}")
    print(f"Duração Total: {duration:.2f} segundos")
    print(f"Total de Amostras: {total_samples}")
    print(f"Taxa Média: {total_samples/duration:.1f} amostras/s")
    print("-" * 30)
    print("ESTATÍSTICAS DE CLASSE:")
    for cls, count in class_counts.items():
        perc = (count / total_samples) * 100
        conf = avg_confidence.get(cls, 0)
        print(f"  - {cls}: {count} amostras ({perc:.1f}%) | Confiança Média: {conf:.1f}%")
    print("-" * 30)
    print("ANÁLISE DE ALERTAS:")
    print(f"  - Alertas Confirmados (Alta Confiança): {real_alerts}")
    print(f"  - Jammers Ignorados (Baixa Confiança): {ignored_alerts}")
    print(f"  - Alertas Suprimidos (Cooldown): {cooldown_events}")
    
    if real_alerts > 1:
        alert_times = df[df['alert_status'] == 'ALERTA_JAMMER']['elapsed_time']
        intervals = alert_times.diff().dropna()
        print(f"  - Intervalo Médio entre Alertas: {intervals.mean():.2f} s")
    
    print("="*50)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        try:
            files = [f for f in os.listdir('.') if f.startswith('log_detector') and f.endswith('.csv')]
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                filename = files[0]
            else:
                filename = "log_detector_detalhado_20251118_195643.csv"
        except Exception as e:
            print(f"Erro ao listar arquivos: {e}")
            sys.exit(1)
            
    analyze_detector_log(filename)