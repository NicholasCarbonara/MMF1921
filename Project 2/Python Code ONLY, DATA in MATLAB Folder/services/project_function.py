from services.advanced_strategy import AdvancedStrategy

def project_function(periodReturns, periodFactRet, x0=None):
    """
    Main project function that implements the advanced strategy
    
    Parameters:
    -----------
    periodReturns : pandas.DataFrame
        Historical returns
    periodFactRet : pandas.DataFrame
        Historical factor returns
    x0 : numpy.ndarray, optional
        Previous portfolio weights
        
    Returns:
    --------
    numpy.ndarray
        Portfolio weights
    """
    # Initialize the advanced strategy
    strategy = AdvancedStrategy(
        lookback_period=36,  # Use 3 years of data
        max_weight=0.05,     # Maximum 5% in any single asset
        target_vol=0.06      # Target 6% annual volatility
    )
    
    # Execute the strategy
    x = strategy.generate_weights(periodReturns, periodFactRet)
    
    return x
