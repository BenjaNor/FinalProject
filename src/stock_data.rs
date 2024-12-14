use std::collections::HashMap;
use std::error::Error;
use csv::ReaderBuilder;

#[derive(Debug)]
pub struct StockData {
    pub ticker: String,
    pub year: u32,
    pub assets: f64,
    pub cash: f64,
    pub equity: f64,
    pub profit: f64,
    pub revenue: f64,
    pub price_change: f64, // Yearly price change
    pub profit_margin: f64, // Profit margin
    pub roa: f64,           // Return on assets
    pub change_in_revenue: Option<f64>, // Change in revenue over the previous year
    pub change_in_profit_margin: Option<f64>, // Change in profit margin over the previous year
    pub change_in_roa: Option<f64>,           // Change in ROA over the previous year
}

pub fn process_stock_data(
    financial_files: &[(&str, &str)],
    price_file: &str,
) -> Result<HashMap<String, Vec<StockData>>, Box<dyn Error>> {

    fn read_csv(file_path: &str) -> Result<HashMap<String, HashMap<u32, f64>>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new().from_path(file_path)?;
        let mut data: HashMap<String, HashMap<u32, f64>> = HashMap::new();

        for result in reader.records() {
            let record = result?;
            let ticker = record.get(0).unwrap_or("").to_string();
            if ticker.is_empty() {
                continue;
            }
            let mut years = HashMap::new();
            for (i, value) in record.iter().skip(1).enumerate() {
                let year = 2022 - i as u32;
                let value: f64 = value.parse().unwrap_or(0.0);
                years.insert(year, value);
            }
            data.insert(ticker, years);
        }
        Ok(data)
    }


    fn calculate_price_changes(file_path: &str) -> Result<HashMap<String, HashMap<u32, f64>>, Box<dyn Error>> {
        let mut reader = ReaderBuilder::new().from_path(file_path)?;
        let headers = reader.headers()?.clone();
        let mut data: HashMap<String, HashMap<u32, Vec<(u32, f64)>>> = HashMap::new();

        for result in reader.records() {
            let record = result?;
            let date = record.get(1).unwrap_or("");
            if date.len() < 7 {
                continue;
            }

            let year: u32 = date[..4].parse().unwrap_or(0);
            let month: u32 = date[5..7].parse().unwrap_or(0);

            for (i, header) in headers.iter().enumerate().skip(2) {
                let ticker = header.to_string();
                let price: f64 = record.get(i).unwrap_or("0").parse().unwrap_or(0.0);

                data.entry(ticker.clone())
                    .or_insert_with(HashMap::new)
                    .entry(year)
                    .or_insert_with(Vec::new)
                    .push((month, price));
            }
        }

        let mut price_changes: HashMap<String, HashMap<u32, f64>> = HashMap::new();

        for (ticker, years) in &data {
            let mut changes = HashMap::new();
            for (year, prices) in years {
                let mut first_month_prices = Vec::new();
                let mut last_month_prices = Vec::new();

                for &(month, price) in prices {
                    if month <= 2 {
                        first_month_prices.push(price);
                    } else if month >= 11 {
                        last_month_prices.push(price);
                    }
                }

                if !first_month_prices.is_empty() && !last_month_prices.is_empty() {
                    let first_avg: f64 = first_month_prices.iter().sum::<f64>() / first_month_prices.len() as f64;
                    let last_avg: f64 = last_month_prices.iter().sum::<f64>() / last_month_prices.len() as f64;
                    let percent_change = ((last_avg - first_avg) / first_avg) * 100.0;
                    changes.insert(*year, percent_change);
                }
            }
            price_changes.insert(ticker.clone(), changes);
        }
        Ok(price_changes)
    }


    let price_changes = calculate_price_changes(price_file)?;
    let assets = read_csv(financial_files[0].0)?;
    let cash = read_csv(financial_files[1].0)?;
    let equity = read_csv(financial_files[2].0)?;
    let profit = read_csv(financial_files[3].0)?;
    let revenue = read_csv(financial_files[4].0)?;

    // Combine datasets
    let mut combined_data: HashMap<String, Vec<StockData>> = HashMap::new();

    for (ticker, years) in &assets {
        let mut stock_data = Vec::new();

        for (&year, &asset_value) in years {
            let cash_value = cash.get(ticker).and_then(|y| y.get(&year)).cloned().unwrap_or(0.0);
            let equity_value = equity.get(ticker).and_then(|y| y.get(&year)).cloned().unwrap_or(0.0);
            let profit_value = profit.get(ticker).and_then(|y| y.get(&year)).cloned().unwrap_or(0.0);
            let revenue_value = revenue.get(ticker).and_then(|y| y.get(&year)).cloned().unwrap_or(0.0);
            let price_change = price_changes
                .get(ticker)
                .and_then(|y| y.get(&year))
                .cloned()
                .unwrap_or(0.0);

            let profit_margin = if revenue_value != 0.0 {
                profit_value / revenue_value
            } else {
                0.0
            };

            let roa = if asset_value != 0.0 {
                (profit_margin * revenue_value) / asset_value
            } else {
                0.0
            };

            stock_data.push(StockData {
                ticker: ticker.clone(),
                year,
                assets: asset_value,
                cash: cash_value,
                equity: equity_value,
                profit: profit_value,
                revenue: revenue_value,
                price_change,
                profit_margin,
                roa,
                change_in_revenue: None,
                change_in_profit_margin: None,
                change_in_roa: None,
            });
        }

        for i in 1..stock_data.len() {
            let (prev, current) = stock_data.split_at_mut(i);
            let prev = &prev[i - 1];
            let current = &mut current[0];

            current.change_in_revenue = Some(current.revenue - prev.revenue);
            current.change_in_profit_margin = Some(current.profit_margin - prev.profit_margin);
            current.change_in_roa = Some(current.roa - prev.roa);
        }

        combined_data.insert(ticker.clone(), stock_data);
    }

    Ok(combined_data)
}
