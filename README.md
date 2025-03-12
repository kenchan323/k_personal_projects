# kenchan323 Personal Projects

### General Infrastructure Build-out (WIP)
- [last updated - 2025-03-12]
    - A simple backtesting engine
    - data loader(s) for financial market data
    - simple ETL pipeline using Arctic DB
    - more to come...

### correlation_matrix
- [last updated - 2019]
    - Performing correlation clustering on Dow Jones stock returns then visualise clustering
    using tsne (t-distributed stochastic neighbouring) technique.
    
### lda_nlp
- [last updated - 2019]
    - Training a LDA (Latent Dirichlet Allocation) model on a corpus of ~2500 BBC News articles from 2004/2005.
    Using LDA we cna detect k number of topics (k being a hyper-parameter) and see the "weights" of words in each
    of these k topics.
    LDA is a topic modelling technique within the domain of Natural Lanaguage Processing.

### portfolio_construction
- [last updated - 2021]
    - script_risk_budget.py: To solve for portfolio risk budgeting optimal solutions using both convex and non-convex 
    problems. It can be shown that the former gives a more optimal solution than the latter (as a global minima is
    guaranteed to be found in a convex problem). 
    - script_max_diversification.py: To solve for a MDP (maximum diversification portfolio) using scipy optimiser.

### senator_trade_backtest
- [last updated - 2020]
    - Using Selenium Chrome webdriver, scrape down all the submitted trades of US senator from the US Senate financial
    disclousre webpage. Then carry out event study to understand subsequent performance of stocks over a certain horizon
    after each Senator trade was carried out (using yfinance - Yahoo Finance API as pricing data source). So effectively 
    treating each Senator's trade as a long/short signal.

### web_scraper / fsa_rating (WIP)
- [last updated - 2019]
    - Scrape (using Selenium with Chrome driver) FSA (Food Standards Agency) hygiene ratings on takeaways restaurants
    within proximity of a postcode based on JustEat listings.
    
### web_scraper / levis_web_check (WIP)
- [last updated - 2019]
    - Continuous price scraping (using Selenium with Chrome driver) on selected item listings on the Levis UK e-commerce 
    store. Once a price drop (or change) is detected, an email alert is sent to the user.