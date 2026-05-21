# Business Report

## 1. Executive Answer
The analysis identified key predictors of student dropout and academic underperformance, enabling targeted interventions. The best-performing model, a Gradient Boosting Classifier, achieved an F1 weighted score of 0.825, indicating a robust ability to classify students at risk. 

## 2. Key Evidence
The dataset comprised 76,518 students, with three target classes: Graduate, Dropout, and Enrolled. The distribution of the target classes was as follows:
- Graduate: 29,025
- Dropout: 20,237
- Enrolled: 11,952

Key features influencing student success included:
- **Curricular units approved in the second semester** (correlation: 0.783): A strong indicator of student retention, suggesting that students who successfully complete more units are less likely to drop out.
- **Tuition fees up to date** (correlation: 0.415): Students with current tuition payments are more likely to graduate, indicating financial stability as a critical factor.
- **Scholarship holder** (correlation: 0.395): Receiving a scholarship positively impacts student retention, highlighting the importance of financial support.

## 3. Model Performance
The selected model for predicting academic risk was the Gradient Boosting Classifier, which demonstrated the following performance metrics:
- **Accuracy**: 82.77%
- **F1 Weighted Score**: 0.825

These metrics indicate that the model effectively balances precision and recall, making it suitable for identifying at-risk students.

## 4. Business Recommendations
1. **Implement Early Intervention Programs**: Focus on students with low curricular unit approvals, particularly in the second semester. Offer tutoring and counseling to improve their academic performance.
2. **Enhance Financial Support**: Increase awareness and accessibility of scholarships and financial aid programs, as financial stability significantly correlates with student success.
3. **Monitor Tuition Payment Status**: Develop a system to proactively engage with students who fall behind on tuition payments, providing them with resources to maintain their enrollment status.

## 5. Limitations and Next Steps
While the analysis provides valuable insights, it is based on synthetic data, which may not fully capture real-world complexities. Future steps should include:
- **Validation with Real Data**: Test the model on actual student data to assess its effectiveness in a real-world context.
- **Continuous Monitoring**: Regularly update the model with new data to adapt to changing student demographics and behaviors.
- **Explore Additional Features**: Investigate other potential predictors, such as mental health indicators or extracurricular involvement, to enhance the model's predictive power.