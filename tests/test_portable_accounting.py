import numpy as np
import pandas as pd
import pytest
from portable_runner.strategy import SimpleMovingAverageStrategy

def frame(prices, dates=None):
    return pd.DataFrame({'Adj Close': prices}, index=dates if dates is not None else pd.date_range('2024-01-01', periods=len(prices)))

def test_buy_preserves_equity_and_records_actual_shares():
    result = SimpleMovingAverageStrategy(1, 2, 1000, 0).run_backtest({'A': frame([100,110,120,130])}, False)
    assert result['portfolio_value'].iloc[2] == pytest.approx(1000)
    assert result['positions']['A'].iloc[2] == pytest.approx(1000/120)
    assert result['portfolio_value'].iloc[3] == pytest.approx(1000*130/120)

def test_flat_prices_do_not_generate_profit():
    result = SimpleMovingAverageStrategy(1, 2).run_backtest({'A': frame([100]*10)}, False)
    assert not result['trades']
    assert (result['portfolio_value'] == 100000).all()

def test_costs_reduce_equity_without_negative_cash():
    result = SimpleMovingAverageStrategy(1, 2, 1000, 10).run_backtest({'A': frame([100,110,120])}, False)
    assert result['portfolio_value'].iloc[-1] == pytest.approx(1000/1.001)
    assert result['cash'].min() >= -1e-8

def test_missing_quote_does_not_erase_holding():
    dates = pd.date_range('2024-01-01', periods=4)
    result = SimpleMovingAverageStrategy(1, 2, 1000, 0).run_backtest({'A': frame([100,110,120], dates[:3]), 'B': frame([100]*4, dates)}, False)
    assert result['positions']['A'].iloc[-1] == result['positions']['A'].iloc[-2]
    assert result['portfolio_value'].iloc[-1] == pytest.approx(1000)

def test_future_price_cannot_change_earlier_equity():
    strategy = SimpleMovingAverageStrategy(1, 2, 1000, 0)
    a = strategy.run_backtest({'A': frame([100,110,120,130])}, False)
    b = strategy.run_backtest({'A': frame([100,110,120,1])}, False)
    pd.testing.assert_series_equal(a['portfolio_value'].iloc[:3], b['portfolio_value'].iloc[:3])

@pytest.mark.parametrize('prices', [[], [0], [np.nan], [-1]])
def test_invalid_prices_fail_explicitly(prices):
    with pytest.raises(ValueError):
        SimpleMovingAverageStrategy(1, 2).run_backtest({'A': frame(prices)}, False)
