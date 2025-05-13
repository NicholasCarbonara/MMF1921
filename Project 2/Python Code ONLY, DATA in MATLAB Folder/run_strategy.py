import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gmean
from services.project_function import project_function

def main():
    print("Loading data...")
    # Load data
    adjClose = pd.read_csv("MMF1921_AssetPrices_3.csv", index_col=0)
    factorRet = pd.read_csv("MMF1921_FactorReturns_3.csv", index_col=0)
    
    # Convert index to datetime
    adjClose.index = pd.to_datetime(adjClose.index)
    factorRet.index = pd.to_datetime(factorRet.index)
    
    # Initial parameters
    initialVal = 100000  # Initial investment
    investPeriod = 3    # Investment period in months
    
    # Extract risk-free rate and factor returns
    riskFree = factorRet['RF']
    factorRet = factorRet.loc[:, factorRet.columns != 'RF']
    
    # Get tickers and dates
    tickers = adjClose.columns
    dates = factorRet.index
    
    # Calculate returns
    returns = adjClose.pct_change(1).iloc[1:, :]
    returns = returns - np.diag(riskFree.values) @ np.ones_like(returns.values)
    adjClose = adjClose.iloc[1:,:]
    
    # Verify data alignment
    assert adjClose.index[0] == returns.index[0]
    assert adjClose.index[0] == factorRet.index[0]
    
    print("Setting up backtest parameters...")
    # Set up backtest parameters
    testStart = returns.index[0] + pd.offsets.DateOffset(years=5)
    testEnd = testStart + pd.offsets.DateOffset(months=investPeriod) - pd.offsets.DateOffset(days=1)
    calEnd = testStart - pd.offsets.DateOffset(days=1)
    
    # Calculate number of periods
    NoPeriods = math.ceil((returns.index[-1].to_period('M') - testStart.to_period('M')).n / investPeriod)
    n = len(tickers)
    
    # Preallocate arrays
    x = np.zeros([n, NoPeriods])
    x0 = np.zeros([n, NoPeriods])
    currentVal = np.zeros([NoPeriods, 1])
    turnover = np.zeros([NoPeriods, 1])
    
    print("Starting backtest...")
    start_time = time.time()
    portfValue = []
    
    for t in range(NoPeriods):
        print(f"Processing period {t+1}/{NoPeriods}")
        
        # Get data for current period
        periodReturns = returns[returns.index <= calEnd]
        periodFactRet = factorRet[factorRet.index <= calEnd]
        
        current_price_idx = (calEnd - pd.offsets.DateOffset(months=1) <= adjClose.index) & (adjClose.index <= calEnd)
        currentPrices = adjClose[current_price_idx]
        
        periodPrices_idx = (testStart <= adjClose.index) & (adjClose.index <= testEnd)
        periodPrices = adjClose[periodPrices_idx]
        
        # Update portfolio value
        if t == 0:
            currentVal[0] = initialVal
        else:
            currentVal[t] = currentPrices @ NoShares.values.T
            x0[:,t] = currentPrices.values * NoShares.values / currentVal[t]
        
        # Get portfolio weights
        x[:,t] = project_function(periodReturns, periodFactRet, x0[:,t] if t > 0 else None)
        
        # Volatility targeting (target 12% annualized)
        portf_vol = np.sqrt(np.dot(x[:,t].T, np.dot(periodReturns.cov().values * 12, x[:,t])))
        target_vol = 0.12
        if portf_vol > 0:
            x[:,t] = x[:,t] * (target_vol / portf_vol)
        # Ensure weights sum to 1 (normalize)
        x[:,t] = x[:,t] / np.sum(np.abs(x[:,t]))
        
        # Calculate turnover
        if t > 0:
            turnover[t] = np.sum(np.abs(x[:,t] - x0[:,t]))
        
        # Calculate number of shares
        NoShares = x[:,t] * currentVal[t] / currentPrices
        
        # Store portfolio value
        portfValue.append(periodPrices @ NoShares.values.T)
        
        # Update dates
        testStart = testStart + pd.offsets.DateOffset(months=investPeriod)
        testEnd = testStart + pd.offsets.DateOffset(months=investPeriod) - pd.offsets.DateOffset(days=1)
        calEnd = testStart - pd.offsets.DateOffset(days=1)
    
    # Combine portfolio values
    portfValue = pd.concat(portfValue, axis=0)
    end_time = time.time()
    
    # Save results to CSV
    portfValue.to_csv('portfolio_value_results.csv')
    
    # Calculate performance metrics
    # Compute periodic (monthly) returns
    portf_returns = portfValue.pct_change().dropna()
    # Align risk-free rate to portfolio return dates
    rf_aligned = riskFree.loc[portf_returns.index]
    excess_returns = portf_returns.values.flatten() - rf_aligned.values
    # Annualize mean and std dev
    mean_excess_return_annual = np.mean(excess_returns) * 12
    std_excess_return_annual = np.std(excess_returns, ddof=1) * np.sqrt(12)
    sharpe_ratio = mean_excess_return_annual / std_excess_return_annual if std_excess_return_annual > 0 else np.nan
    total_return = (portfValue.iloc[-1] / initialVal - 1) * 100
    annual_return = (1 + total_return/100) ** (12/len(portfValue)) - 1
    annual_vol = portfValue.pct_change().std() * np.sqrt(12) * 100
    avg_turnover = np.mean(turnover[1:]) * 100  # Exclude first period
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {float(total_return):.2f}%")
    print(f"Annual Return: {float(annual_return)*100:.2f}%")
    print(f"Annual Volatility: {float(annual_vol):.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Average Turnover: {float(avg_turnover):.2f}%")
    print(f"Runtime: {end_time - start_time:.2f} seconds")
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfValue.index, portfValue.values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main() 