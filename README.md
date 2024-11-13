# Análise Preditiva de Rotatividade de Funcionários

## Descrição

O objetivo deste projeto é analisar a rotatividade de funcionários em uma empresa de alocação de recursos humanos (RH) e desenvolver um modelo preditivo para identificar quais fatores contribuem para a saída de funcionários. A partir da análise dos dados de RH, buscamos fornecer insights sobre os fatores mais influentes na rotatividade e ajudar as empresas a adotar estratégias para reduzir o turnover, criando um ambiente de trabalho mais saudável e uma maior retenção de talentos.

## Problema de Negócio

Uma empresa de alocação de RH tem enfrentado críticas devido à alta taxa de turnover dos funcionários em empresas contratantes. A empresa deseja entender os fatores que mais influenciam a rotatividade de funcionários e desenvolver um modelo capaz de prever se um funcionário específico deixará a empresa ou não. O objetivo final é melhorar as estratégias de retenção de funcionários e ajudar as empresas a se tornarem mais bem-sucedidas.

## Solução Proposta

Para resolver o problema de rotatividade de funcionários, desenvolvemos um modelo analítico usando dados históricos de turnover. O projeto utiliza o arquivo "Dados_RH_Turnover.csv" contendo várias variáveis relacionadas ao perfil dos funcionários e suas experiências na empresa, como:

- Satisfação no trabalho
- Avaliações de desempenho
- Número de projetos realizados
- Anos de trabalho na empresa
- Entre outras características

Foi utilizado um conjunto de algoritmos de classificação para construir e testar o modelo preditivo:

- Árvore de Decisão
- K-Nearest Neighbors (K-NN)
- Naive Bayes
- Regressão Logística
- Redes Neurais

Além disso, o projeto compara as acurácias e matrizes de confusão dos modelos para identificar o melhor desempenho. A implementação pode ser feita utilizando a ferramenta **Orange Datamining** ou **Python**.

## Tecnologias Utilizadas

- **Python** (Bibliotecas: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`)

## Instalação

 **Python**: Para rodar o código em Python, certifique-se de ter o Python instalado. Em seguida, instale as bibliotecas necessárias utilizando o comando abaixo:

   ```bash
   pip install pandas matplotlib seaborn scikit-learn
   ```
   
