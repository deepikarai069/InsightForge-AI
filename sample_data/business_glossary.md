# Revenue
Revenue is the net sales amount of an order after discounts: quantity × unit_price × (1 − discount). It excludes tax and shipping. The `revenue` column in the orders table is the authoritative revenue figure.

# Average Order Value (AOV)
AOV is total revenue divided by the number of distinct orders (order_id). Use it to assess basket size and pricing effects.

# Gross Margin
Gross margin is revenue minus cost of goods sold. Cost of goods sold is quantity × unit_cost from the products table (join on product_id). Gross margin % is gross margin divided by revenue.

# Return Rate
Return rate is the share of orders where returned = 1, i.e. AVG(returned). Apparel typically has the highest return rate.

# Active Customer
An active customer is a customer who placed at least one order in the period under analysis. Count DISTINCT customer_id in orders.

# Customer Segment
Segments are Consumer, Small Business and Enterprise. Enterprise customers order less frequently but have larger basket values.

# Channel
Channel is the sales route: Online (web store), Retail (physical stores) or Partner (resellers). Online is the largest channel.

# Region
Sales regions are North, South, East and West. West is the largest region by revenue.
