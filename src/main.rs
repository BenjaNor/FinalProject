mod stock_data;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use stock_data::process_stock_data;

fn prepare_dataset(
    stock_data: &std::collections::HashMap<String, Vec<stock_data::StockData>>,
) -> (DenseMatrix<f64>, Vec<u8>) {
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (_, records) in stock_data {
        for i in 1..records.len() {
            let current = &records[i];
            let previous = &records[i - 1];

            if i == 1
                || previous.change_in_revenue.is_none()
                || previous.change_in_profit_margin.is_none()
                || previous.change_in_roa.is_none()
            {
                continue;
            }

            let delta_revenue = current.change_in_revenue.unwrap();
            let delta_profit_margin = current.change_in_profit_margin.unwrap();
            let delta_roa = current.change_in_roa.unwrap();

            let current_cash_to_assets = if current.assets != 0.0 {
                current.cash / current.assets
            } else {
                0.0
            };
            let previous_cash_to_assets = if previous.assets != 0.0 {
                previous.cash / previous.assets
            } else {
                0.0
            };
            let delta_cash_to_assets = current_cash_to_assets - previous_cash_to_assets;

            let current_equity_to_assets = if current.assets != 0.0 {
                current.equity / current.assets
            } else {
                0.0
            };
            let previous_equity_to_assets = if previous.assets != 0.0 {
                previous.equity / previous.assets
            } else {
                0.0
            };
            let delta_equity_to_assets = current_equity_to_assets - previous_equity_to_assets;

            features.push(vec![
                delta_revenue,
                delta_profit_margin,
                delta_roa,
                delta_cash_to_assets,
                delta_equity_to_assets,
                delta_revenue * delta_profit_margin, // Interaction
            ]);

            labels.push(categorize_price_change(current.price_change));
        }
    }

    let feature_matrix = DenseMatrix::from_2d_vec(&features);

    (feature_matrix, labels)
}

fn categorize_price_change(price_change: f64) -> u8 {
    match price_change {
        pc if pc < -50.0 => 0,
        pc if pc < 0.0 => 1,
        pc if pc < 50.0 => 2,
        pc if pc > 50.0 => 3,
        _ => 3,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let financial_files = vec![
        ("data_assets.csv", "assets"),
        ("data_cash.csv", "cash"),
        ("data_equity.csv", "equity"),
        ("data_profit.csv", "profit"),
        ("data_revenue.csv", "revenue"),
    ];
    let stock_data = process_stock_data(&financial_files, "stock_prices.csv")?;

    let (features, labels) = prepare_dataset(&stock_data);

    let (x_train, x_test, y_train, y_test) =
        train_test_split(&features, &labels, 0.8, true, None);

    let rf_params = RandomForestClassifierParameters {
        n_trees: 500,
        max_depth: Some(10),
        min_samples_split: 25,
        m: Some(3), 
        ..Default::default()
    };
    let rf_classifier = RandomForestClassifier::fit(&x_train, &y_train, rf_params)?;

    let y_pred = rf_classifier.predict(&x_test)?;

    let acc = accuracy(&y_test, &y_pred);
    println!("Random Forest Classifier Accuracy: {:.2}%", acc * 100.0);

    Ok(())
}

#[cfg(test)]
#[test]
fn test_categorize_price_change() {
    assert_eq!(categorize_price_change(-60.0), 0); 
    assert_eq!(categorize_price_change(-30.0), 1);
    assert_eq!(categorize_price_change(10.0), 2);  
    assert_eq!(categorize_price_change(70.0), 3); 
}

#[cfg(test)]
use super::*;
#[test]
    fn test_change_calculations() {
        let mut stock_data = std::collections::HashMap::new();
        stock_data.insert(
            "FAKE".to_string(),
            vec![
                StockData {
                    ticker: "FAKE".to_string(),
                    year: 2021,
                    assets: 100.0,       
                    cash: 50.0,
                    equity: 30.0,
                    profit: 20.0,
                    revenue: 100.0,
                    price_change: 0.0,
                    profit_margin: 0.2,  // 20 / 100
                    roa: 0.2,            // (0.2 * 100) / 100
                    change_in_revenue: None,
                    change_in_profit_margin: None,
                    change_in_roa: None,
                },
                StockData {
                    ticker: "FAKE".to_string(),
                    year: 2022,
                    assets: 200.0,
                    cash: 100.0,
                    equity: 60.0,
                    profit: 50.0,
                    revenue: 200.0,
                    price_change: 0.0,
                    profit_margin: 0.25, // 50 / 200
                    roa: 0.25,           // (0.25 * 200) / 200
                    change_in_revenue: None,
                    change_in_profit_margin: None,
                    change_in_roa: None,
                },
            ],
        );

        let (features, _) = prepare_dataset(&stock_data);

        let row = features.get_row(0).unwrap();
        assert_eq!(row[0], 100.0); 
        assert_eq!(row[1], 0.05);  
        assert_eq!(row[2], 0.05); 
    }
}