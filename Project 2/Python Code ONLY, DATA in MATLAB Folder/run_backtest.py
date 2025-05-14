import pandas as pd
import numpy as np
from services.advanced_strategy import AdvancedStrategy

def calculate_metrics(portfolio_values):
    """Calculate key performance metrics"""
    returns = portfolio_values.pct_change().dropna()
    
    # Total Return
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    
    # Annual Return
    years = len(returns) / 12
    annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    excess_returns = returns - 0.02/12  # Monthly excess returns
    sharpe_ratio = np.sqrt(12) * excess_returns.mean() / returns.std()
    
    # Calculate turnover (average monthly turnover)
    monthly_turnover = returns.abs().mean() * 12 * 100  # Annualized and in percentage
    
    return {
        'Total Return (%)': round(total_return, 2),
        'Annual Return (%)': round(annual_return, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Turnover (%)': round(monthly_turnover, 2)
    }

def main():
    # Load data
    try:
        # Load asset prices and convert to returns
        asset_prices = pd.read_csv('MMF1921_AssetPrices_3.csv')
        asset_prices['Date'] = pd.to_datetime(asset_prices['Date'])
        asset_prices.set_index('Date', inplace=True)
        returns = asset_prices.select_dtypes(include=[np.number]).pct_change().dropna()
        
        # Load factor returns
        factor_returns = pd.read_csv('MMF1921_FactorReturns_3.csv')
        factor_returns['Date'] = pd.to_datetime(factor_returns['Date'])
        factor_returns.set_index('Date', inplace=True)
        factors = factor_returns.select_dtypes(include=[np.number])
        
        # Ensure data alignment
        common_dates = returns.index.intersection(factors.index)
        returns = returns.loc[common_dates]
        factors = factors.loc[common_dates]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize strategy with optimized parameters
    strategy = AdvancedStrategy(
        lookback_period=36,      # 3-year lookback
        max_weight=0.05,         # 5% position limit
        target_vol=0.12,         # 12% target volatility
        rebalance_freq=12,       # Annual rebalancing
        turnover_penalty=10.0    # High turnover penalty
    )
    
    # Run backtest
    portfolio_values = []
    portfolio_weights = []
    portfolio_returns = []
    current_weights = None
    
    # Initial portfolio value
    portfolio_value = 100
    portfolio_values.append(portfolio_value)
    
    # Track metrics
    total_turnover = 0
    n_rebalances = 0
    
    print("\nRunning backtest...")
    
    # Run through each period
    for t in range(len(returns)):
        if t < strategy.lookback_period:
            continue
            
        # Get historical data up to time t
        hist_returns = returns.iloc[:t+1]
        hist_factors = factors.iloc[:t+1]
        
        # Generate new weights
        new_weights = strategy.generate_weights(hist_returns, hist_factors)
        
        # Calculate turnover
        if current_weights is not None:
            turnover = np.sum(np.abs(new_weights - current_weights))
            total_turnover += turnover
            if turnover > 0:
                n_rebalances += 1
        
        # Update weights
        current_weights = new_weights
        portfolio_weights.append(current_weights)
        
        # Calculate period return
        period_return = np.sum(current_weights * returns.iloc[t])
        portfolio_returns.append(period_return)
        
        # Update portfolio value
        portfolio_value *= (1 + period_return)
        portfolio_values.append(portfolio_value)
        
        # Print progress
        if t % 12 == 0:
            print(f"Processing month {t}/{len(returns)}")
    
    # Convert to arrays/series
    portfolio_values = np.array(portfolio_values)
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate performance metrics
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    annual_return = (1 + total_return/100) ** (12/len(portfolio_returns)) - 1
    annual_return *= 100
    
    volatility = np.std(portfolio_returns) * np.sqrt(12) * 100
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    avg_turnover = total_turnover / (n_rebalances if n_rebalances > 0 else 1)
    
    # Print results
    print("\nBacktest Results:")
    print("=" * 50)
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annual Return: {annual_return:.2f}%")
    print(f"Volatility: {volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Average Turnover: {avg_turnover:.2f}")
    print(f"Number of Rebalances: {n_rebalances}")
    print("=" * 50)

if __name__ == "__main__":
    main() 