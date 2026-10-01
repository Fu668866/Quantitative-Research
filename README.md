# Quantitative Research
I’m a newcomer to quantitative finance, working on some simple strategies for practice. Comments and discussions are welcome—let’s learn together! I’d also like to ask the pros here: any recommendations for quant projects and job-hunting advice?
I use JoinQuant platform to write strategies. Because it is convenient to gain market and financial data from this platform that is based on Python.

# "joinquant_multifactor_basic.py" is a very simple strategy!
In this project, I built a multi-factor stock selection strategy on the JoinQuant platform using historical market and financial data. Constructed a stock universe from historical CSI 300 constituents and incorporated four factors: low price-to-book ratio, small market capitalization, 60-day momentum, and low volatility. Applied cross-sectional percentile ranking to standardize factor scores, combined them using equal weighting, and selected the top 10 stocks for equal-weighted investment. Conducted monthly portfolio rebalancing while accounting for realistic pricing, transaction costs, and look-ahead bias prevention.
