import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from io import BytesIO

# Função para calcular retornos diários
def calculate_daily_returns(data):
    return data['Close'].pct_change().dropna()

# Carregar os dados das ações
def load_data(file_path):
    return pd.read_excel(file_path)

# Carregar os dados de cada ação
data_elet3 = load_data('ELET3_year1.xlsx')
data_vale3 = load_data('VALE3_year1.xlsx')
data_petr4 = load_data('PETR4_year1.xlsx')
data_mglu3 = load_data('MGLU3_year1.xlsx')
data_bbdc4 = load_data('BBDC4_year1.xlsx')

# Calcular os retornos diários de cada ação
returns_elet3 = calculate_daily_returns(data_elet3)
returns_vale3 = calculate_daily_returns(data_vale3)
returns_petr4 = calculate_daily_returns(data_petr4)
returns_mglu3 = calculate_daily_returns(data_mglu3)
returns_bbdc4 = calculate_daily_returns(data_bbdc4)

# Criar DataFrame com preços e retornos diários
prices = pd.DataFrame({
    'Data': data_elet3['Date'],
    'Preço ELET3': data_elet3['Close'],
    'Preço VALE3': data_vale3['Close'],
    'Preço PETR4': data_petr4['Close'],
    'Preço MGLU3': data_mglu3['Close'],
    'Preço BBDC4': data_bbdc4['Close']
})

returns = pd.DataFrame({
    'Data': data_elet3['Date'],
    'Retorno ELET3': returns_elet3,
    'Retorno VALE3': returns_vale3,
    'Retorno PETR4': returns_petr4,
    'Retorno MGLU3': returns_mglu3,
    'Retorno BBDC4': returns_bbdc4
}).set_index('Data')

# Estimar a matriz de covariância e a matriz de correlação
cov_matrix = returns.cov()
correlation_matrix = returns.corr()

# Calcular os retornos esperados
expected_returns = returns.mean()

# Função para calcular risco da carteira
def calculate_portfolio_risk(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Função para calcular retorno da carteira
def calculate_portfolio_return(weights, expected_returns):
    return np.sum(expected_returns * weights)

# Gerar carteiras aleatórias e calcular seus retornos e riscos
num_portfolios = 1000
results = np.zeros((num_portfolios, 3))
num_assets = len(expected_returns)

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    portfolio_return = calculate_portfolio_return(weights, expected_returns) * 252
    portfolio_risk = calculate_portfolio_risk(weights, cov_matrix) * np.sqrt(252)

    results[i, 0] = portfolio_return
    results[i, 1] = portfolio_risk
    results[i, 2] = portfolio_return / portfolio_risk

# Portfólio de Máximo Sharpe Ratio
def negative_sharpe_ratio(weights, expected_returns, cov_matrix):
    p_return = calculate_portfolio_return(weights, expected_returns)
    p_risk = calculate_portfolio_risk(weights, cov_matrix)
    return -p_return / p_risk

def optimize_portfolio(expected_returns, cov_matrix, minimize_risk=True):
    num_assets = len(expected_returns)
    args = (cov_matrix,)

    if minimize_risk:
        def risk(weights, cov_matrix):
            return calculate_portfolio_risk(weights, cov_matrix)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))

        result = minimize(
            fun=risk,
            x0=np.ones(num_assets) / num_assets,
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

    else:
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(num_assets))

        result = minimize(
            fun=negative_sharpe_ratio,
            x0=np.ones(num_assets) / num_assets,
            args=(expected_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x

# Calcular as carteiras de mínimo risco e máximo Sharpe Ratio
weights_min_variance = optimize_portfolio(expected_returns, cov_matrix, minimize_risk=True)
weights_max_sharpe = optimize_portfolio(expected_returns, cov_matrix, minimize_risk=False)

# Retorno e risco dos portfólios de mínimo risco e máximo Sharpe Ratio
min_variance_return = calculate_portfolio_return(weights_min_variance, expected_returns) * 252
min_variance_risk = calculate_portfolio_risk(weights_min_variance, cov_matrix) * np.sqrt(252)

max_sharpe_return = calculate_portfolio_return(weights_max_sharpe, expected_returns) * 252
max_sharpe_risk = calculate_portfolio_risk(weights_max_sharpe, cov_matrix) * np.sqrt(252)

# Plotar a fronteira eficiente
plt.figure(figsize=(10, 6))
plt.scatter(results[:, 1], results[:, 0], c=results[:, 2], cmap='YlGnBu', marker='o')
plt.title('Fronteira Eficiente de Markowitz')
plt.xlabel('Risco (Desvio-Padrão)')
plt.ylabel('Retorno Esperado')
plt.colorbar(label='Índice de Sharpe')

# Destacar os portfólios de mínimo risco e máximo Sharpe Ratio
plt.scatter(min_variance_risk, min_variance_return, color='red', marker='*', s=200, label='Portfólio Mínimo Risco')
plt.scatter(max_sharpe_risk, max_sharpe_return, color='blue', marker='*', s=200, label='Portfólio Máximo Sharpe Ratio')

plt.legend()
plt.grid(True)

# Salvar o gráfico em um buffer de bytes
img = BytesIO()
plt.savefig(img, format='png')
img.seek(0)
plt.close()

# Salvar os resultados em um arquivo Excel
with pd.ExcelWriter('analysis_results.xlsx', engine='openpyxl') as writer:
    # Exportar preços históricos
    prices.to_excel(writer, sheet_name='Preços Históricos', index=False)

    # Exportar retornos diários
    returns.to_excel(writer, sheet_name='Retornos Diários')

    # Exportar excesso de retorno
    excess_returns = returns.sub(returns.mean(), axis=1)
    excess_returns.to_excel(writer, sheet_name='Excesso de Retorno')

    # Exportar matrizes
    cov_matrix.to_excel(writer, sheet_name='Matriz de Covariância')
    correlation_matrix.to_excel(writer, sheet_name='Matriz de Correlação')

    # Exportar resultados dos portfólios
    portfolios = pd.DataFrame({
        'Portfólio': ['Equally Weighted', 'Min Variance', 'Max Sharpe'],
        'Peso BBDC4': [0.20, weights_min_variance[0], weights_max_sharpe[0]],
        'Peso ELET3': [0.20, weights_min_variance[1], weights_max_sharpe[1]],
        'Peso HAPV3': [0.20, weights_min_variance[2], weights_max_sharpe[2]],
        'Peso RENT3': [0.20, weights_min_variance[3], weights_max_sharpe[3]],
        'Peso LREN3': [0.20, weights_min_variance[4], weights_max_sharpe[4]],
        'Retorno Esperado (a.a.)': [9.1577, min_variance_return * 100, max_sharpe_return * 100],
        'Desvio-Padrão': [2.5907, min_variance_risk * 100, max_sharpe_risk * 100],
        'Índice de Sharpe': [1.3424, min_variance_return / min_variance_risk, max_sharpe_return / max_sharpe_risk]
    })
    portfolios.to_excel(writer, sheet_name='Resultados dos Portfólios', index=False)

    # Adicionar gráfico
    workbook = writer.book
    worksheet = workbook.create_sheet(title='Fronteira Eficiente')
    img.seek(0)
    image = Image(img)
    worksheet.add_image(image, 'A1')

print("\nResultados salvos em 'analysis_results.xlsx' com o gráfico da Fronteira Eficaz.")

