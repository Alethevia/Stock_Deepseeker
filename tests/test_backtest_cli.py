from datetime import datetime
import pandas as pd
import pytest
from scripts.run_backtest import fetch_data, generate_mock_data, print_results

def test_real_data_failure_never_silently_uses_simulation(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr('scripts.run_backtest.YahooFinanceProvider.get_historical_prices', fail)
    with pytest.raises(RuntimeError, match="--mock"):
        fetch_data(['AAPL'], datetime(2023, 1, 1), datetime(2023, 2, 1))

def test_mock_is_reproducible_without_changing_global_rng():
    import numpy as np
    np.random.seed(42)
    expected = np.random.random()
    np.random.seed(42)
    a = generate_mock_data('AAPL', datetime(2023, 1, 1), datetime(2023, 2, 1))
    assert np.random.random() == expected
    b = generate_mock_data('AAPL', datetime(2023, 1, 1), datetime(2023, 2, 1))
    pd.testing.assert_frame_equal(a.data, b.data)

def test_cli_displays_actual_engine_event_statistics(capsys):
    print_results({'statistics': {'market_events': 130, 'fill_events': 4}})
    output = capsys.readouterr().out
    assert '市场事件: 130' in output
    assert '成交事件: 4' in output


def test_v2_strategy_places_orders_and_finalizes_holdings():
    from src.backtest.engine_v2 import BacktestEngineV2, BacktestConfig, Strategy
    from src.backtest.events import SignalEvent
    from src.data.providers.base import PriceData
    class BuyOnce(Strategy):
        def generate_signals(self, date, data, portfolio):
            if date == datetime(2023, 1, 3):
                return [SignalEvent(timestamp=date, symbol='A', signal_type='LONG', strength=.1)]
            return []
    frame = pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=100000.),
                         index=pd.bdate_range('2023-01-02', '2023-01-13'))
    config = BacktestConfig(datetime(2023,1,2),datetime(2023,1,13),warmup_period=0)
    engine = BacktestEngineV2(config)
    engine.load_data({'A': PriceData('A',frame,config.start_date,config.end_date)})
    engine.set_strategy(BuyOnce('smoke'))
    result = engine.run()
    assert result['statistics']['fill_events'] >= 1
    assert result['portfolio']['final_value'] < config.initial_capital
    assert all(p.quantity == 0 for p in engine.portfolio.positions.values())
    assert result['equity_curve']['total_value'].iloc[-1] == pytest.approx(result['portfolio']['final_value'])
    assert result['equity_curve'].index.is_unique
