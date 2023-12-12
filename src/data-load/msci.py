from msci.sdk.calc.portfolio import mos
import pandas as pd

# 1. Connect to the MOS API

session = mos.MOSSession(client_id='YOUR KEY', client_secret='YOUR SECRET')
session.ping()

# 2. Load a portfolio using tax lots
sample_data = pd.DataFrame([{"openTradeDate": "2016-12-30", "ISIN": "US02079K3059", "quantity": 1000, "openCostBasis": 792.45, "Asset Name": "ALPHABET INC"},
                            {"openTradeDate": "2016-12-30", "ISIN": "US0231351067", "quantity": 450, "openCostBasis": 749.87, "Asset Name": "AMAZON.COM INC"},
                            {"openTradeDate": "2016-12-30", "ISIN": "US30303M1027", "quantity": 900, "openCostBasis": 115.05, "Asset Name": "FACEBOOK INC"}])

portfolio = session.upload_taxlot_portfolio(
    portfolio_id='MyTaxLotPortfolio', as_of_date='2021-12-31', asset_id='ISIN', taxlot_df=sample_data)

# 3. Run the rebalance using the preferred strategy
mytemplate = mos.TaxAdvantagedModelTrackingTemplate(analysis_date='2022-01-03',portfolio=portfolio)

# 4. Executing the MSCI Optimizer session
job = session.execute(profile=mytemplate)

# Wait for the results to come back
job.wait()

# 5. Show the portfolio level results
job.get_valuations()

# Fetching the optimizer result
opt_result = job.optimizer_result()

# Portfolio Summary
opt_result.get_portfolio_summary_detail()

# Display portfolio and trade suggestions
job.rebalanced_portfolio_on('2022-01-03')
job.trade_suggestions_on('2022-01-03')