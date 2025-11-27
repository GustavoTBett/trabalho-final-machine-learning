# 🏥 Sistema de Predição de Cura da Tuberculose

Aplicação web desenvolvida com Streamlit para predição de cura de tuberculose utilizando modelos de Machine Learning.

## 📋 Descrição

Este sistema utiliza dois modelos de Machine Learning para prever a probabilidade de cura de pacientes com tuberculose:
- **Regressão Logística**: Modelo linear probabilístico
- **Árvore de Decisão**: Modelo baseado em regras de decisão

## 🚀 Funcionalidades

- ✅ Comparação de desempenho entre modelos
- 🔮 Interface para fazer predições em tempo real
- 📊 Análise exploratória dos dados
- 📈 Visualizações de métricas e matriz de confusão
- 📥 Download dos dados processados

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/GustavoTBett/trabalho-final-machine-learning.git
cd trabalho-final-machine-learning
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Como Usar

1. Execute a aplicação:
```bash
streamlit run streamlit_app.py
```

2. Acesse no navegador: `http://localhost:8501`

3. Navegue pelas páginas:
   - **Início**: Visão geral do sistema
   - **Comparação de Modelos**: Métricas e visualizações comparativas
   - **Fazer Predição**: Interface para predição de novos casos
   - **Análise dos Dados**: Exploração do dataset

## 📦 Dependências

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📊 Dataset

O sistema utiliza o arquivo `dados_tuberculose.csv` contendo informações de casos de tuberculose.

## 🎯 Métricas Avaliadas

- Acurácia
- Precisão
- Revocação (Recall)
- F1-Score
- Matriz de Confusão

### 📊 Resultados dos Modelos

#### Árvore de Decisão ⭐ (Recomendado)
- **Acurácia**: 78.64%
- **Precisão**: 81.33%
- **Revocação**: 85.72%
- **F1-Score**: 83.47%

**Matriz de Confusão:**
- VN: 1886 | FP: 944
- FN: 685 | VP: 4113

**Por que é recomendado?**
- ✅ Melhor identificação de casos de cura (maior revocação)
- ✅ Menos falsos negativos (crucial em contexto médico)
- ✅ Melhor F1-Score (equilíbrio entre precisão e revocação)

#### Regressão Logística
- Modelo alternativo mais conservador
- Melhor para identificar casos que NÃO serão curados
- Menos falsos positivos

## 👥 Autores

Desenvolvido como trabalho final da disciplina de Machine Learning.

## 📄 Licença

Este projeto está sob a licença especificada no arquivo LICENSE.
