# ============================================================
# 聚宽基础版：A 股多因子选股策略
# ============================================================
# 研究主线：沪深 300 股票池 -> 估值/动量/低波动/规模因子
#         -> 截面排名 -> 选择最高分股票 -> 月度调仓
#
# 使用方式：将本文件全部复制到聚宽“策略研究/回测”的策略编辑器中运行。
# 数据方式：行情和财务数据由聚宽平台在回测日期提供，不需要本地下载。
# 研究边界：这是教学版策略，未加入行业中性、停牌成交模拟和复杂风险模型。

from jqdata import *
import numpy as np
import pandas as pd


# ============================================================
# 1. 策略初始化：设置基准、交易成本和调仓规则
# ============================================================
def initialize(context):
    # 使用沪深 300 作为基准，便于判断策略是否跑赢大盘蓝筹股票。
    set_benchmark('000300.XSHG')

    # 使用真实价格模式，避免回测下单价格被系统简化为理想价格。
    set_option('use_real_price', True)

    # 开启防止未来函数的检查。
    set_option('avoid_future_data', True)

    # 设置股票交易成本：卖出印花税、买卖佣金和最低佣金。
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type='stock',
    )

    # 保存策略参数；先选择 10 只股票，方便新手理解和观察。
    g.stock_num = 10

    # 每月第一个交易日开盘后调仓一次，降低换手率和交易成本。
    run_monthly(rebalance, monthday=1, time='09:35')

    # 每日收盘前记录组合规模，方便在聚宽图表中检查策略运行情况。
    run_daily(record_portfolio, time='14:50')


# ============================================================
# 2. 股票池：沪深 300 + 基本交易状态过滤
# ============================================================
def get_candidate_stocks(context):
    # 使用调仓日前一天的沪深 300 成分股，避免使用未来成分股信息。
    index_stocks = get_index_stocks('000300.XSHG', date=context.previous_date)

    # 读取调仓时点可观察到的交易状态。
    current_data = get_current_data()

    # 过滤停牌、ST、退市整理期和名称中带有退字的股票。
    candidates = []
    for stock in index_stocks:
        info = current_data[stock]
        if info.paused:
            continue
        if info.is_st:
            continue
        if '退' in info.name:
            continue
        candidates.append(stock)

    return candidates


# ============================================================
# 3. 因子计算：价值、动量、低波动和规模
# ============================================================
def calculate_factors(context, stocks):
    # 在调仓日前获取历史财务数据，确保财务因子不使用未来数据。
    fundamental_query = query(
        valuation.code,
        valuation.pb_ratio,
        valuation.market_cap,
        indicator.roe,
    ).filter(
        valuation.code.in_(stocks),
        valuation.pb_ratio > 0,
        valuation.market_cap > 0,
        indicator.roe > 0,
    )

    fundamentals = get_fundamentals(
        fundamental_query,
        date=context.previous_date,
    )

    # 没有完整财务数据时，无法可靠计算价值和规模因子。
    if fundamentals.empty:
        return pd.DataFrame()

    factor_rows = []

    # 对每只股票计算价格类因子。
    for stock in fundamentals['code']:
        # 获取截至调仓日前一天的 61 个交易日复权收盘价。
        prices = get_price(
            stock,
            end_date=context.previous_date,
            count=61,
            frequency='daily',
            fields=['close'],
            fq='pre',
            skip_paused=True,
        )

        # 少于 61 个价格点时，无法稳定计算 60 日因子，跳过该股票。
        if prices is None or len(prices) < 61:
            continue

        # 计算过去 60 个交易日的动量收益率。
        momentum = prices['close'].iloc[-1] / prices['close'].iloc[0] - 1

        # 计算每日收益率，再计算 60 日年化波动率。
        daily_returns = prices['close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # 读取当前股票的财务因子记录。
        fundamental_row = fundamentals[fundamentals['code'] == stock].iloc[0]

        # 保存原始因子；PB 和市值将在后面取对数并反向排名。
        factor_rows.append(
            {
                'code': stock,
                'pb_ratio': fundamental_row['pb_ratio'],
                'market_cap': fundamental_row['market_cap'],
                'roe': fundamental_row['roe'],
                'momentum': momentum,
                'volatility': volatility,
            }
        )

    factors = pd.DataFrame(factor_rows)

    # 没有可用价格因子时返回空表，避免后续错误下单。
    if factors.empty:
        return factors

    # 价值因子：PB 越低，股票相对越便宜，因此取 -log(PB)。
    factors['value_factor'] = -np.log(factors['pb_ratio'])

    # 规模因子：市值越小，规模因子得分越高，因此取 -log(市值)。
    factors['size_factor'] = -np.log(factors['market_cap'])

    # 动量因子直接使用过去 60 日收益率，收益越高得分越高。
    factors['momentum_factor'] = factors['momentum']

    # 低波动因子将波动率取负，波动率越低得分越高。
    factors['low_volatility_factor'] = -factors['volatility']

    # 删除任何因子缺失或不为有限数值的股票。
    factor_columns = [
        'value_factor',
        'size_factor',
        'momentum_factor',
        'low_volatility_factor',
    ]
    factors = factors.replace([np.inf, -np.inf], np.nan)
    factors = factors.dropna(subset=factor_columns)

    return factors


# ============================================================
# 4. 因子合成：截面排名并形成综合分数
# ============================================================
def make_score(factors):
    # 没有足够股票时不进行排名和交易。
    if factors.empty or len(factors) < g.stock_num:
        return pd.DataFrame()

    # 将每个因子转换为 0 到 1 的截面百分位排名。
    # 排名可以消除 PB、收益率和波动率之间的量纲差异。
    for factor in [
        'value_factor',
        'size_factor',
        'momentum_factor',
        'low_volatility_factor',
    ]:
        factors['{}_rank'.format(factor)] = factors[factor].rank(pct=True)

    # 四个因子等权平均，得到最终选股分数。
    rank_columns = [
        'value_factor_rank',
        'size_factor_rank',
        'momentum_factor_rank',
        'low_volatility_factor_rank',
    ]
    factors['combined_score'] = factors[rank_columns].mean(axis=1)

    # 按综合分数从高到低排序。
    return factors.sort_values('combined_score', ascending=False)


# ============================================================
# 5. 调仓执行：卖出非目标股票，再等权买入目标股票
# ============================================================
def safe_order_target_value(context, stock, target_value, current_data):
    """只在订单至少达到一手时下单，减少聚宽的无效订单提示。"""
    data = current_data[stock]
    price = data.last_price

    # 没有有效价格时无法估算一手股票的金额，直接跳过。
    if price is None or price <= 0:
        return

    position = context.portfolio.positions.get(stock)
    current_value = 0
    if position is not None:
        # position.value 是聚宽提供的当前持仓市值。
        current_value = getattr(position, 'value', 0) or 0

    # 目标市值与当前市值的差额不足 100 股时，不产生无效订单。
    if abs(target_value - current_value) < price * 100:
        return

    # 清仓时必须至少有 100 股可卖，否则聚宽会拒绝订单。
    if target_value <= 0 and position is not None:
        closeable_amount = getattr(position, 'closeable_amount', 0) or 0
        if closeable_amount < 100:
            return

    order_target_value(stock, target_value)


def rebalance(context):
    # 获取经过交易状态过滤的沪深 300 股票池。
    candidates = get_candidate_stocks(context)

    # 计算截至昨日的财务和价格因子。
    factors = calculate_factors(context, candidates)

    # 将原始因子转换成综合排名。
    ranked = make_score(factors)

    # 如果本月数据不足，保留原组合，不进行空交易。
    if ranked.empty:
        log.warn('本月没有足够的有效因子数据，跳过调仓。')
        return

    # 选出综合分数最高的股票。
    target_stocks = list(ranked.head(g.stock_num)['code'])

    # 读取当前持仓，准备计算需要卖出的股票。
    current_stocks = list(context.portfolio.positions.keys())
    current_data = get_current_data()

    # 卖出不再入选且没有封死跌停的股票。
    for stock in current_stocks:
        if stock not in target_stocks:
            if current_data[stock].last_price > current_data[stock].low_limit:
                safe_order_target_value(context, stock, 0, current_data)

    # 将 95% 的组合资产等权分配给目标股票，保留少量现金缓冲。
    target_value = context.portfolio.total_value * 0.95 / len(target_stocks)

    # 买入目标股票；封死涨停时跳过，避免产生不可成交的理想化订单。
    for stock in target_stocks:
        if current_data[stock].last_price < current_data[stock].high_limit:
            safe_order_target_value(context, stock, target_value, current_data)

    # 记录调仓后的股票数量和平均综合分数，便于回测后检查。
    record(target_count=len(target_stocks))
    record(avg_score=float(ranked.head(g.stock_num)['combined_score'].mean()))


# ============================================================
# 6. 记录组合指标：用于聚宽回测图表
# ============================================================
def record_portfolio(context):
    # 记录当前股票持仓数量，观察策略是否长期保持目标分散度。
    record(holdings=len(context.portfolio.positions))

    # 记录当前现金比例，观察是否有大量资金因涨跌停或数据问题未投资。
    if context.portfolio.total_value > 0:
        cash_ratio = context.portfolio.available_cash / context.portfolio.total_value
        record(cash_ratio=cash_ratio)
