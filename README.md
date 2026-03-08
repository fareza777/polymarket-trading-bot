# Polymarket Trading Bot

Advanced trading bot for Polymarket prediction markets featuring:

- **Arbitrage Detection** — Cross-market and intra-market arbitrage scanning
- **Market Making** — Avellaneda-Stoikov based market making with dynamic spreads
- **Risk Management** — Position limits, drawdown protection, correlation-based risk
- **Order Management** — Smart order routing with CLOB API integration
- **Live Dashboard** — Real-time terminal UI for monitoring

## Project Structure

```
polymarket_pro/
├── main.py            # Main entry point & arbitrage bot
├── models.py          # Data models & configurations
├── strategies.py      # Trading strategies (arb + market making)
├── order_manager.py   # Order execution & management
├── risk_manager.py    # Risk management engine
└── dashboard.py       # Terminal dashboard UI
```

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   - `POLY_PRIVATE_KEY`
   - `POLY_API_KEY`
   - `POLY_SECRET`
   - `POLY_PASSPHRASE`

## Disclaimer

This software is for educational purposes only. Trading involves significant risk of loss.
