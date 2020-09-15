if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from hb.utils.date import *
    euro_price_error = pd.read_csv("examples/heston_market/euro_price_error.csv")
    euro_price_error['days']=(360.*euro_price_error.time).astype(np.int)
    euro_price_diff = euro_price_error[euro_price_error.days>=60].relative_error.values
    print(euro_price_diff.mean(), euro_price_diff.std())
    print(euro_price_diff.max(), euro_price_diff.min())
    textstr = '\n'.join((
        r'$\mu=%.4f$' % (euro_price_diff.mean(), ),
        r'$\sigma=%.4f$' % (euro_price_diff.std(), ),
        r'$\min=%.4f$' % (euro_price_diff.min(), ),
        r'$\max=%.4f$' % (euro_price_diff.max(), )))

    fig, ax = plt.subplots()
    ax.hist(euro_price_error.relative_error.values, bins=100, alpha=0.5, range=(-1,1), label='Analytic_MC_Relative_Differences')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    plt.title('European Option Heston Analytic Price vs MC Price Relative Difference')
    plt.show()

    figure = plt.figure()
    plt.scatter(euro_price_error[euro_price_error.days>=60].analytic_price.values, 
                euro_price_error[euro_price_error.days>=60].mc_price.values,
                linewidth=0.3, alpha=0.4, s=0.5)
    plt.xlabel('Analytic Price')
    plt.ylabel('MC Price')
    plt.show()