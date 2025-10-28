from models import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price: float, n_shares: float) -> float:
    """
    Calculate the total portfolio value including cash and positions.

    Parameters:
    cash: Current cash available.
    long_ops: List of long position operations.
    short_ops: List of short position operations.
    current_price: Current price of the asset.
    n_shares: Number of shares per operation.

    Returns:
    Total portfolio value as a float.
    """
    
    port_val = cash

    # add long positions value
    for position in long_ops:
        port_val += current_price * position.n_shares

   # add short positions value
    for position in short_ops:
        pnl = (position.price - current_price) * position.n_shares
        port_val += pnl

    return port_val
