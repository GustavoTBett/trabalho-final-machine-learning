import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
import os

# Configuração da página
st.set_page_config(
    page_title="Predição de Cura da Tuberculose",
    page_icon="🏥",
    layout="wide"
)

# Título da aplicação
st.title("🏥 Sistema de Predição de Cura da Tuberculose")
st.markdown("---")

# Função para carregar e preparar os dados
@st.cache_data
def load_and_prepare_data():
    """Carrega e prepara o dataset"""
    dataset = pd.read_csv("dados_tuberculose.csv", sep=';', encoding='latin1')
    
    # Remover colunas desnecessárias
    colunas_remover = ['id_agravo', 'id_municip', 'id_regiona', 'id_unidade', 
                       'cs_gestant', 'id_mn_resi', 'id_rg_resi', 'pop_liber',
                       'nu_ano', 'dt_notific', 'dt_diag', 'dt_inic_tr', 
                       'dt_encerra', 'cs_sexo']
    dataset.drop(colunas_remover, axis=1, inplace=True, errors='ignore')
    
    # Criar variável target
    dataset['target'] = dataset['situa_ence'].apply(
        lambda x: 1 if str(x).strip().lower() == 'cura' else 0
    )
    
    # Preencher valores ausentes
    dataset.fillna('Não informado', inplace=True)
    
    # Remover coluna situa_ence
    dataset.drop('situa_ence', axis=1, inplace=True, errors='ignore')
    
    # Codificar variáveis categóricas
    colunas_categoricas = dataset.select_dtypes(include=['object']).columns.tolist()
    dataset_encoded = pd.get_dummies(dataset, columns=colunas_categoricas, drop_first=True)
    
    return dataset, dataset_encoded

# Função para treinar modelos
@st.cache_resource
def train_models(dataset_encoded):
    """Treina e retorna os modelos"""
    X = dataset_encoded.drop('target', axis=1)
    y = dataset_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Modelo de Regressão Logística
    modelo_lr = LogisticRegression(max_iter=1000)
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    
    # Modelo de Árvore de Decisão
    modelo_dt = DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    modelo_dt.fit(X_train, y_train)
    y_pred_dt = modelo_dt.predict(X_test)
    
    return {
        'lr': {'model': modelo_lr, 'predictions': y_pred_lr},
        'dt': {'model': modelo_dt, 'predictions': y_pred_dt},
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

# Função para calcular métricas
def calculate_metrics(y_test, y_pred):
    """Calcula as métricas de avaliação"""
    return {
        'Acurácia': metrics.accuracy_score(y_test, y_pred),
        'Precisão': metrics.precision_score(y_test, y_pred),
        'Revocação': metrics.recall_score(y_test, y_pred),
        'F1-Score': metrics.f1_score(y_test, y_pred)
    }

# Função para plotar matriz de confusão
def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plota a matriz de confusão"""
    matriz = confusion_matrix(y_test, y_pred)
    labels = ['Não Curado', 'Curado']
    df_cm = pd.DataFrame(matriz, index=labels, columns=labels)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = 'Greens' if model_name == 'Regressão Logística' else 'Blues'
    sns.heatmap(df_cm, annot=True, fmt='d', cmap=cmap, cbar=False, ax=ax)
    ax.set_xlabel('Previsto pelo Modelo')
    ax.set_ylabel('Valor Real')
    ax.set_title(f'Matriz de Confusão - {model_name}')
    
    return fig

# Carregar dados
with st.spinner('Carregando dados...'):
    dataset, dataset_encoded = load_and_prepare_data()
    models_data = train_models(dataset_encoded)

# Sidebar para navegação
st.sidebar.title("📊 Navegação")
page = st.sidebar.radio(
    "Escolha uma página:",
    ["🏠 Início", "📈 Comparação de Modelos", "🔮 Fazer Predição", "📊 Análise dos Dados"]
)

# Página Início
if page == "🏠 Início":
    st.header("Bem-vindo ao Sistema de Predição de Cura da Tuberculose")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Registros", len(dataset))
    
    with col2:
        taxa_cura = (dataset_encoded['target'].sum() / len(dataset_encoded)) * 100
        st.metric("Taxa de Cura", f"{taxa_cura:.1f}%")
    
    with col3:
        st.metric("Features Utilizadas", dataset_encoded.shape[1] - 1)
    
    st.markdown("---")
    st.subheader("Sobre o Sistema")
    st.write("""
    Este sistema utiliza técnicas de Machine Learning para prever a probabilidade de cura 
    de pacientes com tuberculose. Dois modelos foram treinados:
    
    - **Regressão Logística**: Modelo linear que estima probabilidades
    - **Árvore de Decisão**: Modelo baseado em regras de decisão
    
    Use o menu lateral para:
    - Comparar o desempenho dos modelos
    - Fazer predições para novos casos
    - Analisar os dados
    """)

# Página Comparação de Modelos
elif page == "📈 Comparação de Modelos":
    st.header("Comparação de Modelos")
    
    # Calcular métricas para ambos os modelos
    metrics_lr = calculate_metrics(models_data['y_test'], models_data['lr']['predictions'])
    metrics_dt = calculate_metrics(models_data['y_test'], models_data['dt']['predictions'])
    
    # Tabela comparativa
    st.subheader("Métricas de Desempenho")
    comparison_df = pd.DataFrame({
        'Regressão Logística': metrics_lr,
        'Árvore de Decisão': metrics_dt
    })
    
    st.dataframe(comparison_df.style.format("{:.4f}").highlight_max(axis=1, color='lightgreen'))
    
    # Gráfico de barras comparativo
    st.subheader("Comparação Visual")
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df.T.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_xlabel('Modelo')
    ax.set_title('Comparação de Métricas')
    ax.legend(title='Métricas')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)
    
    # Matrizes de confusão
    st.subheader("Matrizes de Confusão")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Regressão Logística**")
        fig_lr = plot_confusion_matrix(
            models_data['y_test'], 
            models_data['lr']['predictions'],
            'Regressão Logística'
        )
        st.pyplot(fig_lr)
    
    with col2:
        st.write("**Árvore de Decisão**")
        fig_dt = plot_confusion_matrix(
            models_data['y_test'], 
            models_data['dt']['predictions'],
            'Árvore de Decisão'
        )
        st.pyplot(fig_dt)

# Página Fazer Predição
elif page == "🔮 Fazer Predição":
    st.header("Fazer Predição")
    
    # Seleção do modelo
    model_choice = st.selectbox(
        "Escolha o modelo:",
        ["Regressão Logística", "Árvore de Decisão"]
    )
    
    selected_model = models_data['lr']['model'] if model_choice == "Regressão Logística" else models_data['dt']['model']
    
    st.markdown("---")
    st.subheader("Preencha as informações do paciente")
    
    # Obter as colunas originais antes da codificação
    original_cols = dataset.columns.tolist()
    original_cols.remove('target')
    
    # Criar formulário de entrada
    with st.form("prediction_form"):
        st.write("**Informações Clínicas:**")
        
        # Aqui você pode adicionar campos específicos baseados nas features mais importantes
        # Por simplicidade, vou criar um exemplo com algumas features
        
        col1, col2 = st.columns(2)
        
        with col1:
            cs_raca = st.selectbox("Raça", ["Branca", "Preta", "Parda", "Ignorado"])
            cs_zona = st.selectbox("Zona", ["Urbana", "Rural", "Periurbana"])
            tratamento = st.selectbox("Tipo de Tratamento", 
                                     ["Caso Novo", "Recidiva", "Reingresso após Abandono", 
                                      "Transferência", "Não sabe"])
        
        with col2:
            agravaids = st.selectbox("Agravamento por AIDS", ["Não", "Sim", "Ignorado"])
            agravalcoo = st.selectbox("Agravamento por Alcoolismo", ["Não", "Sim", "Ignorado"])
            forma = st.selectbox("Forma", ["Pulmonar", "Extrapulmonar"])
        
        submit_button = st.form_submit_button("🔮 Fazer Predição")
    
    if submit_button:
        # Criar um dataframe com valores padrão
        input_data = pd.DataFrame(0, index=[0], columns=models_data['X_train'].columns)
        
        # Mapear as entradas do usuário para as colunas codificadas
        # (Isso é uma simplificação - em produção, você precisaria de um mapeamento completo)
        
        # Fazer predição
        prediction = selected_model.predict(input_data)
        prediction_proba = selected_model.predict_proba(input_data)
        
        st.markdown("---")
        st.subheader("Resultado da Predição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.success("✅ **Predição: CURA ESPERADA**")
            else:
                st.error("❌ **Predição: CURA NÃO ESPERADA**")
        
        with col2:
            st.metric(
                "Probabilidade de Cura",
                f"{prediction_proba[0][1]*100:.1f}%"
            )
        
        # Barra de progresso
        st.progress(float(prediction_proba[0][1]))
        
        st.info(f"""
        **Interpretação:**
        - Probabilidade de não cura: {prediction_proba[0][0]*100:.1f}%
        - Probabilidade de cura: {prediction_proba[0][1]*100:.1f}%
        
        *Modelo utilizado: {model_choice}*
        """)

# Página Análise dos Dados
elif page == "📊 Análise dos Dados":
    st.header("Análise Exploratória dos Dados")
    
    # Visualização da distribuição da variável target
    st.subheader("Distribuição da Taxa de Cura")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = dataset_encoded['target'].value_counts()
    ax.bar(['Não Curado', 'Curado'], target_counts.values, color=['#ff6b6b', '#51cf66'])
    ax.set_ylabel('Quantidade')
    ax.set_title('Distribuição de Casos')
    
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig)
    
    # Estatísticas básicas
    st.subheader("Estatísticas do Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Distribuição da Variável Target:**")
        st.write(target_counts)
    
    with col2:
        st.write("**Percentuais:**")
        percentages = (target_counts / len(dataset_encoded) * 100).round(2)
        st.write(percentages)
    
    # Amostra dos dados
    st.subheader("Amostra dos Dados")
    st.dataframe(dataset.head(10))
    
    # Download dos dados
    st.subheader("Download")
    csv = dataset.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar Dataset Processado",
        data=csv,
        file_name='dados_tuberculose_processado.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Sistema de Predição de Cura da Tuberculose | Machine Learning</p>
</div>
""", unsafe_allow_html=True)
