# Relatório de Aprendizado de Máquina - Análise de resultados

O presente relatório tem por objetivo analisar os resultados do fine tuning de hiper parâmetros para os 5 algoritmos e modelos de aprendizado de máquina propostos pela disciplina, Multi-layer Perceptron (MLP), K-Nearest Neighbors (KNN), Regressão Logística, Árvore de Decisão e Random Forest. A MLP foi implementada em Python usando a API Keras sobre a biblioteca Tensorflow e os demais algoritmos foram implementados em Python usando a biblioteca Scikit-Learn. O modelo de pré-processamento escolhido foi o modelo 2 balanceado que apresentou melhor performance sobre os algoritmos em formato de controle (hiper parâmetros definidos pelos professores) conforme descrito no relatório de pré-processamento. 

Os testes foram realizados variando um hiper parâmetro por vez e serão descritos, para cada algoritmo, nas secções subsequentes. As variações foram pensadas a partir do entendimento do funcionamento dos algoritmos e conhecimento sobre as características da base de dados.

### Multi-layer Perceptron (MLP)

O fine tuning de hiper parâmetros para a MLP consistiu em variar a quantidade de epocas, a arquitetura do modelo em termos da quantidade de camadas e quantidade de neurônios por camada, variar a função de ativação e alterar a taxa de aprendizado (loss_weights).

#### Resultados

Melhor arranjo:

### K-Nearest Neighbors (KNN)

Para o KNN foram selecionados dois hiper parâmetros fundamentais, a métrica do cálculo de distância e o número de vizinhos considerados. A forma de medir distância é a forma do algoritmo de medir similaridade entre instâncias e o número de vizinhos implica em quantas instâncias vizinhas serão selecionadas para concluir a classe pela moda tendo relação com o nível de agregação dos casos. Em todos os casos de teste os pesos foram mantidos uniformes por definição dos professores que desejam a implementação do KNN e não do DWNN.

As métricas de distância escolhidas para teste foram a distância de Manhattan, Hassanat e Hamming. A distância de Manhattan foi escolhida por ser uma métrica amplamente utilizada e recomendada para dados categóricos. A distância de Hassanat foi selecionada pela influência do artigo ["Effects of Distance Measure Choice on KNN Classifier Performance - A Review"](https://arxiv.org/pdf/1708.04321) que avaliando 56 métricas de distância sobre 28 datasets de classificação concluiu que a distância de Hassanat retornava a melhor performance. Por fim, a distância de Hamming foi escolhida por ser muito utilizada com dados categóricos, marcando uma comparação direta entre duas cadeias de dígitos.

A ordem de testes consistiu em variar a métrica mantendo o número de vizinhos conforme e controle e, definida a métrica com melhor resultado, variar o número de vizinhos.

#### Resultados

Melhor arranjo:

O GridSearchCV foi utilizado para codificar a combinação de possíveis casos de teste:

### Regressão Logística

Nos testes com o algoritmo de regressão logística, os hiper parâmetros alvo foram o solver, a penalização, o C (parâmetro que dita a força da regularização feita sobre os dados de entrada) e o máximo de iterações (max-iter).

Os solvers escolhidos para teste foram newton-cholesky, sag e liblinear. O solver newton-cholesky foi escolhido por recomendação da própria documentação do Scikit-Learn sobre o classificador de Regressão logística que indica o uso desse solver em bases que apresentam atributos categóricos one-hot encoded com valores de rara aparição, caso frequente na base em questão. Sag foi escolhido por ser otimizado para grandes datasets e o liblinear por ser bom para datasets de muitas dimensões, o que caracteriza o caso do dataset balanceado 2 que apresenta 53 atributos.

A ordem de testes consistiu em variar o solver mantendo os demais hiper parâmetros conforme e controle e, dado o solver com melhor resultado, variar os demais um a um tendo como solver o de melhor desempenho.

#### Resultados

Melhor arranjo:

### Árvore de Decisão

Como a maior desvantagem das árvores de decisão é a susceptibilidade ao overfitting para bases com grande número de instâncias e atributos, como o caso presente, o foco dos testes foi combinar variações de max-depth e min-samples-leaf para que o algoritmo não de foco demasiado às características do conjunto de treinamento, afetando a sua capacidade de generalização e, com isso, a sua acurácia nos casos de teste. A troca do critério de avaliação dos atributos de entropia para gini e log-loss (cross-entropy) também foi explorada.

A ordem de execução dos testes consistiu em primeiro variar o critério e então os demais hiper parâmetros usando o melhor critério observado como padrão.

#### Resultados

Melhor arranjo:

O GridSearchCV foi utilizado para codificar a combinação de possíveis casos de teste:


### Random Forest

Diferentemente da árvore de decisão, a random forest é estruturada de modo a minimizar o overfitting com o conjunto de treinamento e assim o foco do teste consistiu na variação do número de árvores da floresta para captar em diferentes níveis os atributos do dataset. A troca da função para cálculo do max-features de raiz quadrada para log2 e números inteiros definidos também foi explorada, assim como a troca do critério para gini e log-loss.

A ordem de execução dos testes consistiu em primeiro variar o critério, com o melhor critério variar o max-fetures e por fim variar o total de árvores na floresta.

#### Resultados

Melhor arranjo:

O GridSearchCV foi utilizado para codificar a combinação de possíveis casos de teste:

## Conclusão

Pelos resultados pode-se observar que apesar dos extensivos testes, pouca acurácia foi ganha com a variação dos hiper parâmetros, levando a conclusão de que o maior potencial para ganho de performance está na melhora do pré-processamento. Isso, contudo, só pode ser feito em trabalho conjunto com a SESAB, pois há dados faltantes e incoerentes na base original que tiveram que ser abstraídos ou removidos, prejudicando a acurácia dos dados na representação de cenários positivos e negativos.