# Fronteira Eficiente de Markowitz

Este projeto realiza a construção da **fronteira eficiente de Markowitz** para análise de risco-retorno de carteiras de investimento baseadas em ações da bolsa brasileira (B3). Os dados são processados e analisados com auxílio de bibliotecas como `pandas`, `numpy`, `matplotlib` e `scipy`.

## 📊 Objetivo

- Calcular os retornos diários de um conjunto de ações.
- Estimar os riscos (desvio-padrão) e retornos esperados.
- Gerar 1000 carteiras aleatórias para traçar a fronteira eficiente.
- Determinar as carteiras de:
  - **Mínima variância (risco mínimo)**
  - **Máximo índice de Sharpe**
- Exportar resultados para um arquivo Excel com gráficos e métricas.

## 📂 Ações utilizadas

- ELET3 (Eletrobras)
- VALE3 (Vale)
- PETR4 (Petrobras)
- MGLU3 (Magazine Luiza)
- BBDC4 (Bradesco)

> ⚠️ Certifique-se de que os arquivos `.xlsx` contendo os dados estejam no mesmo diretório do script:
> - `ELET3_year1.xlsx`
> - `VALE3_year1.xlsx`
> - `PETR4_year1.xlsx`
> - `MGLU3_year1.xlsx`
> - `BBDC4_year1.xlsx`

## ⚙️ Requisitos

- Python 3.8+
- Bibliotecas:
  - pandas
  - numpy
  - scipy
  - matplotlib
  - openpyxl

Instale as dependências com:

```bash
pip install pandas numpy scipy matplotlib openpyxl
