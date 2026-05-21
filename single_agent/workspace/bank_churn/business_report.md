# Business Report on Customer Churn Analysis

## 1. Executive Answer
The analysis identified key customer segments at high risk of churn, enabling targeted retention strategies. The best-performing model, a Gradient Boosting Classifier, achieved an F1 weighted score of 0.857, indicating strong predictive capability for identifying customers likely to exit.

## 2. Key Evidence
- **Target Variable**: The analysis focused on predicting customer churn, indicated by the binary variable "Exited."
- **Top Features Influencing Churn**:
  - **Age**: Older customers are more likely to churn, with a correlation score of 0.408.
  - **Number of Products**: Customers using fewer products are at higher risk, with a score of 0.340.
  - **Active Membership Status**: Inactive members show a significant churn risk (score of 0.117).
  - **Geography**: Customers from Germany are more likely to churn compared to those from other countries (score of 0.058).
  - **Account Balance**: Lower balances correlate with higher churn risk (score of 0.029).
  
These insights suggest that age, product usage, and active membership status are critical factors in understanding customer retention.

## 3. Model Performance
- **Model Task Type**: Classification
- **Best Model**: Gradient Boosting Classifier
- **Performance Metric**: F1 Weighted Score
- **Test Performance**: 
  - Accuracy: 86.59%
  - F1 Weighted Score: 85.72%

The model's performance indicates a reliable ability to distinguish between customers who are likely to stay and those who are likely to leave.

## 4. Business Recommendations
1. **Targeted Outreach**: Focus retention efforts on older customers and those with fewer products. Personalized communication strategies can be developed to address their specific needs and concerns.
2. **Enhance Product Offerings**: Encourage customers with fewer products to explore additional services that may enhance their banking experience, potentially reducing churn.
3. **Re-engagement Campaigns**: Implement campaigns aimed at inactive members, highlighting the benefits of active membership and available products.
4. **Geographic Focus**: Pay special attention to customers in Germany, as they show a higher propensity to churn. Tailor marketing strategies to address their unique needs and preferences.

## 5. Limitations and Next Steps
While the analysis provides valuable insights, it is important to note that the model's performance may vary with different datasets or in real-world applications. The reliance on synthetic data means that further validation with actual customer data is necessary to confirm these findings. Next steps should include:
- Testing the model on real customer data to validate its predictive power.
- Continuously monitoring churn rates and adjusting strategies based on ongoing analysis.
- Exploring additional features or customer feedback to refine the model and improve retention strategies.