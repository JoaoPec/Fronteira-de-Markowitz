# Fronteira Eficiente de Markowitz

Projeto em Python para analisar carteiras de acoes pela teoria moderna de portfolios. A ferramenta le planilhas historicas, calcula retornos e risco anualizados, simula milhares de carteiras, otimiza os pontos de minima variancia e maximo Sharpe e exporta um relatorio Excel com graficos e metricas.

## O que o projeto entrega

- Descoberta automatica de ativos por padrao de arquivo (`*_year1.xlsx` por padrao).
- Suporte a configuracao manual de ativos via CLI.
- Calculo de retornos diarios, excesso de retorno, matriz de covariancia e matriz de correlacao.
- Carteiras de referencia:
  - igualmente ponderada;
  - minima variancia;
  - maximo indice de Sharpe.
- Simulacao reprodutivel de carteiras aleatorias com seed configuravel.
- Fronteira eficiente otimizada por alvo de retorno.
- Relatorio Excel formatado com abas de resumo, parametros, ativos, series historicas, matrizes e simulacoes.
- Grafico PNG da fronteira eficiente.
- Testes unitarios para as funcoes centrais.

## Estrutura

```text
.
├── main.py                  # CLI e motor de analise
├── requirements.txt         # dependencias Python
├── tests/                   # testes unitarios
├── *_year1.xlsx             # dados historicos de exemplo
├── *_year2.xlsx             # dados historicos adicionais
└── reports/                 # saidas geradas localmente (ignorado pelo git)
```

## Requisitos

- Python 3.10+
- pip

Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

## Como executar

Uso padrao, lendo os arquivos `*_year1.xlsx` no diretorio atual:

```bash
python main.py
```

Saidas geradas:

- `reports/markowitz_analysis.xlsx`
- `reports/efficient_frontier.png`

Rodando com os dados do segundo ano:

```bash
python main.py --pattern "*_year2.xlsx" --output reports/year2_analysis.xlsx --chart reports/year2_frontier.png
```

Configurando ativos manualmente:

```bash
python main.py \
  --asset ELET3=ELET3_year1.xlsx \
  --asset VALE3=VALE3_year1.xlsx \
  --asset PETR4=PETR4_year1.xlsx
```

Alterando parametros da simulacao:

```bash
python main.py --portfolios 10000 --frontier-points 100 --risk-free-rate 0.105 --seed 123
```

## Formato esperado das planilhas

Cada arquivo deve conter pelo menos as colunas:

- `Date`: data do pregao;
- `Close`: preco de fechamento.

As demais colunas podem existir e serao ignoradas pela analise.

## Testes

```bash
python -m unittest discover -s tests
```

## Observacao importante

Este projeto e educacional e nao constitui recomendacao de investimento. Os resultados dependem da janela historica, da qualidade dos dados e das premissas adotadas.
