"""SMA reference backtest using explicit shares and cash, with per-side costs.
Signals execute at the next available adjusted close (a total-return approximation).
"""
from typing import Dict
import numpy as np
import pandas as pd

class SimpleMovingAverageStrategy:
    def __init__(self, short_window=20, long_window=50, initial_capital=100000.0,
                 transaction_cost_bps=5.0):
        if not 0 < short_window < long_window:
            raise ValueError('Require 0 < short_window < long_window')
        if initial_capital <= 0 or not np.isfinite(initial_capital):
            raise ValueError('initial_capital must be finite and positive')
        if not np.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps < 10000:
            raise ValueError('transaction_cost_bps must be in [0, 10000)')
        self.short_window, self.long_window = short_window, long_window
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps

    def run_backtest(self, data: Dict[str, pd.DataFrame], verbose=True) -> Dict:
        if not data:
            raise ValueError('No data provided')
        for symbol, frame in data.items():
            if frame.empty or 'Adj Close' not in frame:
                raise ValueError(f'{symbol}: nonempty data with Adj Close required')
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ValueError(f'{symbol}: index must be unique and sorted')
            prices = frame['Adj Close'].to_numpy(dtype=float)
            if not np.isfinite(prices).all() or (prices <= 0).any():
                raise ValueError(f'{symbol}: prices must be finite and positive')
        symbols = list(data)
        quotes = pd.concat({s: data[s]['Adj Close'] for s in symbols}, axis=1).sort_index()
        marks = quotes.ffill()  # no backfill from future quotes
        signals = {s: self._generate_signals(data[s]) for s in symbols}
        shares = dict.fromkeys(symbols, 0.0)
        positions = {s: pd.Series(0.0, index=quotes.index) for s in symbols}
        portfolio_value = pd.Series(0.0, index=quotes.index)
        cash_history = pd.Series(0.0, index=quotes.index)
        cash = float(self.initial_capital)
        fee_rate = self.transaction_cost_bps / 10000.0
        trades = []
        for date in quotes.index:
            equity = cash + sum(shares[s] * marks.at[date, s] for s in symbols if shares[s] != 0)
            targets = {}
            for s in symbols:
                price = quotes.at[date, s]
                if pd.notna(price):
                    weight = 1.0 / len(symbols) if signals[s].at[date] > 0 else 0.0
                    targets[s] = equity * weight / price
            # Sell first. Missing-price holdings stay intact and use their last mark.
            for s in sorted(targets, key=lambda s: targets[s] - shares[s]):
                price = float(quotes.at[date, s])
                delta = targets[s] - shares[s]
                if delta > 0:
                    delta = min(delta, max(cash, 0.0) / (price * (1 + fee_rate)))
                if abs(delta) < 1e-10:
                    continue
                value = delta * price
                fee = abs(value) * fee_rate
                cash -= value + fee
                shares[s] += delta
                trades.append(dict(date=date, symbol=s, action='BUY' if delta > 0 else 'SELL',
                                   shares=abs(delta), price=price, value=abs(value), transaction_cost=fee))
            for s in symbols:
                positions[s].at[date] = shares[s]
            cash_history.at[date] = cash
            portfolio_value.at[date] = cash + sum(shares[s] * marks.at[date, s] for s in symbols if shares[s] != 0)
        returns = portfolio_value.pct_change(fill_method=None).fillna(0.0)
        if verbose:
            print(f'SMA research backtest: {len(trades)} trades; final equity ${portfolio_value.iloc[-1]:,.2f}; cost {self.transaction_cost_bps:g} bps per side')
        return dict(portfolio_value=portfolio_value, returns=returns, positions=positions,
                    cash=cash_history, trades=trades, symbols=symbols, config=dict(
                        strategy='SMA', short_window=self.short_window, long_window=self.long_window,
                        initial_capital=self.initial_capital, n_symbols=len(symbols),
                        transaction_cost_bps=self.transaction_cost_bps, position_units='shares',
                        execution='next available adjusted close'))

    def _generate_signals(self, df):
        close = df['Adj Close']
        signal = (close.rolling(self.short_window).mean() > close.rolling(self.long_window).mean()).astype(float)
        return signal.shift(1).fillna(0.0)
