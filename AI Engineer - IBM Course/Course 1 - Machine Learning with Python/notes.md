## Detailed Overview

### Module 01: Machine Learning Basics
- Covers supervised (e.g., labeled data for house price prediction) and unsupervised (e.g., customer clustering without labels) learning paradigms.
- Introduces Python libraries:
    - NumPy for array operations (e.g., fast matrix calculations).
    - Pandas for data manipulation (e.g., merging and cleaning datasets).
    - SciPy for scientific computing (e.g., optimization routines).
    - scikit-learn for training, tuning, and evaluating ML models.
    - Matplotlib for visualizing data (e.g., plotting decision boundaries).

### Module 02: Regression
- Aims to predict continuous values (e.g., housing prices, salaries).
- Simple vs. multiple regression:
    - Simple uses one feature (e.g., predicting weight from height).
    - Multiple uses multiple features (e.g., predicting house price from area, bedrooms).
- Linear vs. non-linear relationships:
    - Linear fits a straight line (e.g., y = mx + c).
    - Non-linear uses curves (e.g., polynomial regression).
- Evaluation metrics (MSE, MAE, RMSE) highlight prediction accuracy.
- Overfitting and underfitting guide model selection and regularization.

### Module 03: Classification
- Predicts categorical outcomes (e.g., spam vs. not spam).
- KNN assigns labels based on nearest neighbor majority.
- Decision Tree splits data using metrics like Gini or entropy.
- F1 score and log loss measure classification performance.

### Module 04: Advanced Classification
- Logistic Regression:
    - Suited for binary classification (e.g., pass/fail).
    - Outputs probabilities (e.g., 0–1).
- Support Vector Machine:
    - Uses hyperplanes/kernels for non-linear data (e.g., radial kernel for complex boundaries).
- Multi-class scenarios:
    - Extension of binary classification (e.g., one-vs-rest approach).

### Module 05: Clustering
- Groups unlabeled data by similarity.
- K-means partitions data into K clusters (e.g., customer segmentation).
- Each cluster has a centroid, and points belong to the nearest centroid.

## Example
- Predicting house prices (Regression).
- Classifying emails as spam/ham (Classification).
- Grouping users by behavior (Clustering).
- Testing models with train/test splits to avoid overfitting.

## Interview Q&A

### Module 01 (Machine Learning Basics)
1. Q: What differentiates supervised from unsupervised learning?
    A: Supervised uses labeled data; unsupervised uses unlabeled data.
2. Q: Name a common application of supervised learning.
    A: Predicting house prices.
3. Q: Give an example of unsupervised learning.
    A: Grouping customers based on purchasing patterns.
4. Q: Why are Python libraries important in machine learning?
    A: They simplify data processing and model implementation.
5. Q: What does NumPy specialize in?
    A: Efficient array operations and numerical computations.
6. Q: Why is Pandas useful for data manipulation?
    A: It provides DataFrame structures for easy data cleaning and merging.
7. Q: How does SciPy complement other libraries?
    A: It offers scientific functions like optimization routines.
8. Q: What is scikit-learn best known for?
    A: Standard machine learning algorithms and utilities.
9. Q: Why use Matplotlib?
    A: It helps visualize data and model outputs.
10. Q: How can you avoid confusion in library imports?
     A: Use well-known aliases like “import numpy as np.”

### Module 02 (Regression)
11. Q: What is regression used for?
     A: Predicting continuous values, such as prices.
12. Q: How is simple linear regression different from multiple linear regression?
     A: Simple uses one input variable; multiple uses more than one.
13. Q: When would you choose non-linear regression?
     A: When data shows a curved or complex relationship.
14. Q: Why use MSE in regression?
     A: It penalizes larger errors more heavily.
15. Q: What does MAE measure?
     A: The average magnitude of errors without considering direction.
16. Q: Define RMSE.
     A: It’s the square root of MSE, giving errors in the same units as the target.
17. Q: What is overfitting?
     A: A model that memorizes training data and performs poorly on new data.
18. Q: How do you detect underfitting?
     A: Low accuracy on both training and testing sets.
19. Q: Why split data into training and testing sets?
     A: To assess model performance on unseen data.
20. Q: Give an example of a real-world regression scenario.
     A: Predicting future stock prices based on historical data.

### Module 03 (Classification)
21. Q: What is classification?
     A: Predicting labels for categorical outcomes.
22. Q: How does KNN work?
     A: It assigns a class based on the closest training examples.
23. Q: What is a decision tree?
     A: A tree-based model that splits data using purity measures.
24. Q: Why is F1 score important in classification?
     A: It balances precision and recall into a single metric.
25. Q: Define precision in classification.
     A: The fraction of predicted positives that are actual positives.
26. Q: Define recall in classification.
     A: The fraction of actual positives successfully identified.
27. Q: What is log loss?
     A: A metric indicating how well a model estimates class probabilities.
28. Q: How do you handle class imbalance?
     A: Techniques like oversampling, undersampling, or using specialized metrics.
29. Q: Name a simple model for categorical predictions.
     A: Naive Bayes classifier.
30. Q: Give a practical example of classification.
     A: Classifying emails as spam or not spam.

### Module 04 (Advanced Classification)
31. Q: What kind of problems suit logistic regression?
     A: Binary classification tasks like churn prediction.
32. Q: How do you interpret logistic regression output?
     A: It yields probabilities for each class.
33. Q: Why might you choose SVM over logistic regression?
     A: SVM can handle complex boundaries via kernel functions.
34. Q: What is the role of a hyperplane in SVM?
     A: It separates data into different classes with maximum margin.
35. Q: When would you apply a kernel trick in SVM?
     A: When data is not linearly separable.
36. Q: How do you extend binary classifiers to multi-class?
     A: Approaches include one-vs-rest or one-vs-one.
37. Q: Which kernel might you choose for text classification?
     A: Often the linear kernel is used for high-dimensional data.
38. Q: What parameter in SVM controls regularization?
     A: The “C” parameter.
39. Q: Give a use case for multi-class classification.
     A: Handwriting recognition of digits 0–9.
40. Q: Why compare logistic regression with SVM?
     A: They both handle classification but differ in approach and complexity.

### Module 05 (Clustering)
41. Q: What is clustering?
     A: Grouping similar data points without predefined labels.
42. Q: How does K-means algorithm cluster data?
     A: It divides data into K groups based on distance to centroids.
43. Q: What is a centroid in K-means?
     A: The central point representing each cluster.
44. Q: How do you decide on the optimal K value?
     A: Techniques like the “elbow” method help identify the best K.
45. Q: Why is clustering considered unsupervised?
     A: It does not require labeled data to form groups.
46. Q: What happens if K is too large?
     A: The model may over-segment the data into too many small clusters.
47. Q: What if K is too small?
     A: Distinct subgroups might be forced into broader clusters.
48. Q: How do you handle outliers in K-means?
     A: Consider methods like DBSCAN or robust scaling.
49. Q: Where is clustering often used?
     A: Customer segmentation or image compression.
50. Q: What is the main challenge in clustering?
     A: Determining the appropriate number of clusters and evaluating quality.
