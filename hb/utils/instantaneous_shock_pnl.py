def get_pnl(market):
    portfolio = market.get_portfolio()
    market_counter = market.get_counter()
    total_paths = market_counter.get_total_paths()
    total_steps = market_counter.get_total_steps()
    pnls = []
    market_counter.reset()
    market_counter.inc_step_counter()
    market_counter.inc_path_counter()
    cur_nav = portfolio.get_nav()
    for pi in range(total_paths):
        market_counter._path_counter = pi
        market_counter._step_counter = total_steps
        shocked_nav = portfolio.get_nav()
        pnls.append(shocked_nav - cur_nav)
    return pnls
    