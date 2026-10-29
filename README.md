# Stock Deepseeker — research backtesting code

Experimental strategy, factor, risk and event-driven backtest modules. There is no verified live performance or production-readiness claim.

## Run

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q -o addopts=
python scripts/run_backtest.py --help
```

The portable entry point remains `一键回测_portable.py`; it downloads external prices and may bootstrap dependencies. Its SMA strategy is also directly importable:

```python
from portable_runner.strategy import SimpleMovingAverageStrategy
# data: {symbol: DataFrame}, sorted unique dates, positive finite 'Adj Close'
result = SimpleMovingAverageStrategy().run_backtest(data)
```

## Accounting correction (2026-10-29)

The portable implementation previously mixed portfolio weights with share quantities, corrupting holdings and equity. It now tracks shares and cash separately, carries missing-day holdings at the last observed price, delays signals to the next available adjusted close, and charges an explicit default **5 bps per side**. This is a configurable research assumption, not measured broker execution cost. Adjusted-close execution approximates total returns; it is not a broker fill simulation. Earlier results from the faulty accounting must be rerun.

The package in `src/data/providers/` is the active provider implementation. Its shadowed `providers.py` duplicate and an obsolete generated investor pitch were removed. Tests and current limitations are documented in [AUDIT.md](AUDIT.md).
