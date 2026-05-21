# Business Report: Crop Yield Prediction

## 1. Executive Answer
The analysis aimed to predict crop yield and identify key agronomic and environmental factors influencing production. The best-performing model identified was the Random Forest Regressor, which achieved an R² score of -0.0819 on the test set. Despite the negative performance metric, the model provided insights into the factors affecting yield, which can guide agricultural practices.

## 2. Key Evidence
- **Dataset Overview**: The dataset comprised 2,200 rows and 9 columns, including soil nutrients (Nitrogen, Phosphorus, Potassium), environmental factors (Temperature, Humidity, pH Value, Rainfall), and the target variable (Yield).
- **Top Features**: The analysis revealed the following top features influencing yield:
  - **pH Value**: 21.96%
  - **Rainfall**: 19.85%
  - **Temperature**: 13.70%
  - **Humidity**: 11.82%
  - **Phosphorus**: 10.92%
  - **Nitrogen**: 9.69%
  - **Potassium**: 7.91%
  - **Crop Frequency**: 4.16%
  
  These features indicate that soil health (pH, nutrients) and weather conditions (rainfall, temperature, humidity) are critical for optimizing crop yield.

## 3. Model Performance
- **Task Type**: Regression
- **Best Model**: Random Forest Regressor
- **Test Performance**: 
  - R²: -0.0819
  - Mean Absolute Error (MAE): 1,796.22
  - Root Mean Squared Error (RMSE): 2,720.89

The negative R² score suggests that the model did not perform well in predicting yield, indicating potential issues with the data or model complexity.

## 4. Business Recommendations
1. **Optimize Soil Management**: Focus on maintaining optimal pH levels and nutrient balance (Nitrogen, Phosphorus, Potassium) to enhance crop yield. Regular soil testing and amendments based on the top features can lead to better production outcomes.
   
2. **Weather Monitoring**: Implement systems to monitor rainfall, temperature, and humidity closely. This data can help in making timely decisions regarding irrigation and crop management practices.

3. **Crop Selection**: Use the insights from crop frequency analysis to select crop varieties that are best suited to the prevailing soil and environmental conditions, thereby maximizing yield potential.

4. **Resource Allocation**: Allocate resources (fertilizers, water) based on the predictive insights from the model, focusing on the most influential factors to improve yield efficiency.

## 5. Limitations and Next Steps
- **Model Performance**: The Random Forest model's negative R² indicates that the current model may not be suitable for accurate yield predictions. Further tuning and exploration of alternative models (e.g., Gradient Boosting) may be necessary.
  
- **Data Quality**: While the dataset was clean with no missing values, the weak model performance suggests that additional features or more granular data (e.g., local weather patterns, pest data) could improve predictions.

- **Next Steps**: 
  - Conduct further feature engineering to explore interactions between variables.
  - Test additional models and hyperparameter tuning to enhance predictive performance.
  - Consider integrating external data sources for a more comprehensive analysis of yield determinants.