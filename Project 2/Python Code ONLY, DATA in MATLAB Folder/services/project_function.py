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
        max_weight=0.1,      # Maximum 10% in any single asset
        target_vol=0.15,     # Target 15% annual volatility
        use_pca=True,        # Use PCA for factor reduction
        use_lasso=True,      # Use LASSO for factor selection
        use_shrinkage=True   # Use covariance shrinkage
    )
    
    # Execute the strategy
    x = strategy.execute_strategy(periodReturns, periodFactRet)
    
    return x
