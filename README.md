# Bank Credit Card Type Prediction

Banks and financial institutions offer a variety of credit card products tailored to different customer segments.  
However, identifying the most appropriate credit card type for each customer manually is time-consuming, inefficient, and prone to errors.

The goal of this project is to develop a machine learning-based predictive model that can accurately classify customers into specific credit card categories (e.g., Blue, Silver, Gold, Platinum) based on their demographic, financial, and transactional attributes.

A successful solution to this problem would allow banks to:

- Personalize their credit card offerings based on individual customer profiles.
- Improve customer satisfaction by aligning financial products with customer needs and behaviors.
- Enhance marketing efficiency by targeting the right customers with the right products.
- Reduce customer churn by proactively offering upgrades or incentives.
- Increase overall profitability through better customer segmentation and cross-selling strategies.

In summary, the project aims to leverage historical customer data to automate the prediction process, enabling data-driven decision-making in credit card marketing and customer relationship management.

Building an accurate predictive model will help bridge the gap between customer needs and product offerings, ultimately strengthening customer loyalty and boosting business growth.

![3585404_66249](https://github.com/user-attachments/assets/a02d6636-367a-4414-8a29-97185088026e)


## 🛠️ Project Structure

- `Bank Credit Card Type Prediction.ipynb` — Main notebook containing data analysis, model training, evaluation, and prediction steps.

## 📚 Problem Statement

Banks offer various types of credit cards based on customer profiles. Predicting the most suitable credit card type can help banks improve customer satisfaction and business outcomes. This project builds a classification model to automate this prediction.

## 📊 Dataset Overview

The dataset includes the following features:

- `BankCustomers.csv` — Dataset containing customer demographic and financial details.There are 10127 rows , 21 columns)

- **CLIENTNUM** — Unique customer identification
- **Attrition_Flag** — Indicates if the customer is an existing or attrited customer
- **Customer_Age** — Age of the customer
- **Gender** — Gender of the customer
- **Dependent_count** — Number of dependents
- **Education_Level** — Highest education achieved
- **Marital_Status** — Marital status
- **Income_Category** — Income bracket
- **Card_Category** — (Target variable: e.g., Blue, Silver, Gold, Platinum)
- **Months_on_book** — Duration of relationship with the bank
- **Total_Relationship_Count** — Total number of products held
- **Months_Inactive_12_mon** — Number of inactive months in the last 12 months
- **Contacts_Count_12_mon** — Number of contacts in the last 12 months
- **Credit_Limit** — Credit limit on the card
- **Total_Revolving_Bal** — Total revolving balance on the card
- **Avg_Open_To_Buy** — Average available open-to-buy credit
- **Total_Amt_Chng_Q4_Q1** — Change in transaction amount (Q4 vs Q1)
- **Total_Trans_Amt** — Total transaction amount
- **Total_Trans_Ct** — Total transaction count
- **Total_Ct_Chng_Q4_Q1** — Change in transaction count (Q4 vs Q1)
- **Avg_Utilization_Ratio** — Average card utilization ratio

## 🔍 Data Analysis Process

The complete data analysis and modeling pipeline followed for this project is as below:

1. **Data Collection**  
   - Imported customer data from `BankCustomers.csv`.

2. **Data Understanding**  
   - Reviewed dataset structure, feature types, and sample entries.
   - Summarized statistical information to understand feature distributions.

3. **Exploratory Data Analysis (EDA)**  
   - Checked for missing values, duplicates, and inconsistent entries.
   - Visualized feature distributions using histograms, bar plots, and box plots.
   - Analyzed relationships between features and the target variable (`Card_Category`).
   - Used correlation heatmaps to detect multicollinearity and important feature linkages.

4. **Data Preprocessing**  
   - Handled missing and inconsistent values (if any).
   - Performed Label Encoding and One-Hot Encoding on categorical variables.
   - Scaled numerical features using StandardScaler to normalize feature values.
   - Performed feature engineering based on transaction patterns and credit behavior.

5. **Feature Selection**  
   - Selected important features using correlation matrices and model-based feature importances.
   - Removed redundant or less impactful variables.

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## 🏆 Results

- The **Gradient Boosting Classifier** achieved an impressive **95% accuracy** on the test set, outperforming all other models.
- Decision Tree,RandomForest and SVM showed good potential but were slightly less accurate.

## 📌 Key Insights

- Younger customers (ages 20–35) with higher spending scores tend to prefer **Gold** and **Platinum** credit cards.
- Customers with **high annual income** and **longer work experience** are more likely to opt for **Platinum** cards.
- **Family size** has a minor influence compared to income and profession but shows trends for certain card types.
- **Profession** is a strong indicator — business owners and engineers tend to have premium cards.
- Spending score is **positively correlated** with high-tier cards like Gold and Platinum.
- **Gender** shows a slight preference pattern — male customers leaned more toward Platinum cards.

## 🧠 Conclusion

This project successfully built a classification model to predict credit card types with **95% accuracy** using **Gradient Boosting** . The insights drawn from customer demographics and financial behavior can help banks:

- Personalize credit card offers based on user profiles.
- Improve marketing strategies by focusing on high-income, high-spending customers.
- Identify potential premium customers early based on career and income trends.

With further feature engineering and real-time data integration, the model can be enhanced for even better predictive performance and deployed into production environments.

## 🛠️ Requirements

- Python 3.12
- jupyter Notebook
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

