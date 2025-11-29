import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os

def analyze_session_log(csv_file):
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
    
    # Contagens
    class_counts = df['class_name'].value_counts()
    action_counts = df['action'].value_counts()
    
    # Confiança Média por Classe
    avg_confidence = df.groupby('class_name')['confidence'].mean()
    
    # Análise de Ataques
    attacks = df[df['action'] == 'ATAQUE_INICIADO']
    attacks_started = len(attacks)
    
    # 3. Geração dos Gráficos
    plt.style.use('ggplot') # Estilo visual limpo
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Relatório de Pós-Ação: Jammer Cognitivo\nArquivo: {csv_file}', fontsize=16)
    
    # --- GRÁFICO 1: Linha do Tempo (Timeline) ---
    ax1 = plt.subplot(3, 1, 1)
    
    # Cores para cada classe
    colors = {'Normal (Ocioso)': 'green', 'ALVO: DADOS': 'blue', 'Jammer (Eu/Outro)': 'orange'}
    row_colors = df['class_name'].map(colors).fillna('gray')
    
    # Plota cada ponto de detecção
    ax1.scatter(df['elapsed_time'], df['confidence'], c=row_colors, s=15, alpha=0.6, label='Deteção')
    
    # Marca os momentos de ataque com linhas vermelhas verticais
    for _, row in attacks.iterrows():
        ax1.axvline(x=row['elapsed_time'], color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.text(row['elapsed_time'], 102, 'ATAQUE', color='red', fontsize=9, rotation=90, va='bottom')

    # Legenda manual para ficar claro
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Ocioso', markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='ALVO (Dados)', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Jammer/Ruído', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], color='red', linestyle='--', label='Disparo Realizado')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    ax1.set_ylabel('Confiança da IA (%)')
    ax1.set_xlabel('Tempo decorrido (s)')
    ax1.set_title('Timeline Operacional: Confiança e Engajamentos')
    ax1.set_ylim(0, 115) 
    ax1.grid(True)

    # --- GRÁFICO 2: Pizza (Distribuição do Tempo) ---
    ax2 = plt.subplot(3, 2, 3)
    if not class_counts.empty:
        # Pega as cores correspondentes para manter consistência
        pie_colors = [colors.get(x, 'gray') for x in class_counts.index]
        ax2.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', colors=pie_colors, startangle=140, explode=[0.05]*len(class_counts))
        ax2.set_title('Ocupação do Espectro (Classificação)')
    else:
        ax2.text(0.5, 0.5, 'Sem dados', ha='center')

    # --- GRÁFICO 3: Barras (Ações do Sistema) ---
    ax3 = plt.subplot(3, 2, 4)
    if not action_counts.empty:
        # Filtra apenas ações interessantes para o gráfico
        actions_to_plot = action_counts.drop('MONITORANDO', errors='ignore')
        if not actions_to_plot.empty:
            actions_to_plot.plot(kind='bar', ax=ax3, color='purple', alpha=0.7)
            ax3.set_title('Eventos do Sistema (Excluindo Monitoramento)')
            ax3.set_ylabel('Quantidade')
            ax3.tick_params(axis='x', rotation=0)
            # Adiciona valores nas barras
            for i, v in enumerate(actions_to_plot):
                ax3.text(i, v + 0.1, str(v), ha='center')
        else:
            ax3.text(0.5, 0.5, 'Apenas monitoramento (sem ações)', ha='center')
    
    # --- GRÁFICO 4: Boxplot (Distribuição de Confiança) ---
    ax4 = plt.subplot(3, 1, 3)
    data_to_plot = [df[df['class_name'] == cls]['confidence'] for cls in df['class_name'].unique()]
    labels = df['class_name'].unique()
    
    if len(data_to_plot) > 0:
        ax4.boxplot(data_to_plot, labels=labels, vert=False, patch_artist=True)
        ax4.set_title('Consistência da IA: Distribuição de Confiança por Classe')
        ax4.set_xlabel('Confiança (%)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 4. Salvar e Mostrar
    output_img = csv_file.replace('.csv', '_analise.png')
    plt.savefig(output_img)
    print(f"Gráfico salvo em: {output_img}")
    
    # 5. Relatório de Texto no Terminal
    print("\n" + "="*50)
    print("RELATÓRIO ANALÍTICO FINAL")
    print("="*50)
    print(f"Arquivo: {csv_file}")
    print(f"Duração da Sessão: {duration:.2f} segundos")
    print(f"Total de Amostras: {total_samples}")
    print(f"Taxa de Amostragem: {total_samples/duration:.1f} Hz (amostras/seg)")
    print("-" * 30)
    print("ANÁLISE DE ESPECTRO:")
    for cls, count in class_counts.items():
        perc = (count / total_samples) * 100
        conf = avg_confidence.get(cls, 0)
        print(f"  - {cls}: {count}x ({perc:.1f}%) | Confiança Média: {conf:.1f}%")
    print("-" * 30)
    print("ESTATÍSTICAS DE COMBATE:")
    print(f"  - Ataques Iniciados: {attacks_started}")
    
    cooldown_count = df[df['action'] == 'COOLDOWN'].shape[0]
    print(f"  - Oportunidades Ignoradas (Cooldown): {cooldown_count}")
    
    if attacks_started > 0:
        attack_times = df[df['action'] == 'ATAQUE_INICIADO']['elapsed_time']
        if len(attack_times) > 1:
            intervals = attack_times.diff().dropna()
            print(f"  - Tempo Médio entre Ataques: {intervals.mean():.1f} s")
            print(f"  - Menor Tempo entre Ataques: {intervals.min():.1f} s")
    
    print("="*50)
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        try:
            files = [f for f in os.listdir('.') if f.startswith('log_sessao') and f.endswith('.csv')]
            if files:
                files.sort(key=os.path.getmtime, reverse=True)
                filename = files[0]
            else:
                print("Nenhum arquivo 'log_sessao*.csv' encontrado nesta pasta.")
                sys.exit(1)
        except Exception as e:
             print(f"Erro ao listar arquivos: {e}")
             sys.exit(1)
    
    analyze_session_log(filename)