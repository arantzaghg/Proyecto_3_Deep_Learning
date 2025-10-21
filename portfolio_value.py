from models import Operation

def get_portfolio_value(cash: float, long_ops: list[Operation], short_ops: list[Operation], current_price: float, n_shares: float) -> float:

    port_val = cash

    # add long positions value
    for position in long_ops:
        port_val += current_price * position.n_shares

   # add short positions value
    for position in short_ops:
        pnl = (position.price - current_price) * position.n_shares
        port_val += pnl

    return port_val
