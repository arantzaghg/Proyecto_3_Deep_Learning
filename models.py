from dataclasses import dataclass

@dataclass
class Operation:
    time: any
    price: float
    stop_loss: float
    take_profit: float
    n_shares: int
    type: str 