# Referências Bibliográficas & Apêndice Técnico

Esta página consolida as referências teóricas formais que embasam as decisões tomadas ao longo do projeto, em conformidade com as normas da **ABNT (Associação Brasileira de Normas Técnicas)**, além de fornecer um apêndice de rastreabilidade computacional e instruções para reprodução do ambiente.

---

## 📚 Referências Bibliográficas

1. **BERGSTRA, James; BENGIO, Yoshua.** Random search for hyper-parameter optimization. *Journal of Machine Learning Research (JMLR)*, v. 13, p. 281-305, 2012.  
   *(Referência fundamental que fundamenta a escolha do RandomizedSearchCV frente ao GridSearchCV em espaços de alta dimensão).*

2. **GAMA, João; ŽLIOBAITĖ, Indrė; BIFET, Albert; PECHENIZKIY, Mykola; BOUCHACHIA, Abdelhamid.** A survey on concept drift adaptation. *ACM Computing Surveys (CSUR)*, v. 46, n. 4, p. 1-37, 2014.  
   *(Trabalho seminal sobre a dinâmica temporal e desvios de distribuição estocástica em aprendizado supervisionado).*

3. **HASTIE, Trevor; TIBSHIRANI, Robert; FRIEDMAN, Jerome.** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2. ed. Nova York: Springer-Verlag, 2009.  
   *(Tratado clássico sobre a teoria estatística do aprendizado, regularização Ridge/Lasso, trade-off viés-variância e custos computacionais de ensembles).*

4. **KOHAVI, Ron.** Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. In: *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)*, p. 202-207, 1996.  
   *(Publicação original que introduziu e formalizou a extração do dataset Adult Census Income a partir dos dados do Censo dos EUA de 1994).*

5. **LUNDBERG, Scott M.; LEE, Su-In.** A unified approach to interpreting model predictions. In: *Advances in Neural Information Processing Systems (NeurIPS 2017)*, v. 30, p. 4765-4774, 2017.  
   *(Artigo inaugural que introduziu o framework SHAP unificando os valores de Shapley da teoria dos jogos com explicabilidade aditiva).*

6. **LUNDBERG, Scott M. et al.** From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, v. 2, n. 1, p. 56-67, 2020.  
   *(Artigo que apresentou o algoritmo exato em tempo polinomial TreeExplainer, utilizado para explicar a Random Forest deste projeto).*

7. **MINCER, Jacob.** *Schooling, Experience, and Earnings*. Nova York: National Bureau of Economic Research (NBER), 1974.  
   *(Apresenta a clássica equação de rendimentos de Mincer, justificando econometricamente o impacto exponencial dos anos de estudo e da idade sobre os salários).*

8. **SCULLEY, David et al.** Hidden technical debt in machine learning systems. In: *Advances in Neural Information Processing Systems (NeurIPS 2015)*, v. 28, p. 2503-2511, 2015.  
   *(Obra de referência do Google sobre custos operacionais ocultos, manutenção de modelos, data drift e práticas modernas de MLOps).*

9. **STROBL, Carolin; BOULESTEIX, Anne-Laure; ZEILEIS, Achim; HOTHORN, Torsten.** Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, v. 8, n. 1, p. 25, 2007.  
   *(Identifica o viés intrínseco do critério de impureza de Gini / MDI em favorecer variáveis contínuas ou de alta cardinalidade).*

---

## 🛠️ Apêndice Técnico: Reprodutibilidade e Ambiente

### Resumo dos Melhores Hiperparâmetros

| Modelo / Pipeline | Ferramenta de Otimização | Hiperparâmetros Selecionados | Score de Validação (ROC-AUC) |
| :--- | :--- | :--- | :---: |
| **Regressão Logística (v1)** | `GridSearchCV` | `{'C': 1, 'solver': 'lbfgs'}` | 0,878 |
| **Random Forest (v1)** | `RandomizedSearchCV` | `{'n_estimators': 200, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 20}` | 0,879 |
| **Random Forest Campeã (v2)**| `RandomizedSearchCV` | `{'n_estimators': 300, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 20}` | **0,8884** |

### Como Compilar e Servir esta Documentação Localmente

Para rodar a documentação localmente e visualizar o site interativo no navegador:

```bash
# 1. Ativar o ambiente virtual com mkdocs instalado
source .venv/bin/activate

# 2. Iniciar o servidor local de desenvolvimento
mkdocs serve
```

O servidor disponibilizará a documentação interativa em tempo real no endereço:  
`http://127.0.0.1:8000/`

Para compilar os arquivos estáticos prontos para produção ou deploy no GitHub Pages:
```bash
mkdocs build
```
Os arquivos HTML, CSS e JavaScript serão gerados na pasta estática `site/`.
