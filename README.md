# CPSC-393-final-project-code
This is the code we used to build our models for the final project. We built a KNN, Logistic regression and Decision Tree models, improved them, and then comapred them to a naive baseline model to show the impact of the changes we made to make it more accurate efficient and perform better.

#README.MD
**Project Title: Early Detection of Heart Disease Using Machine Learning
Contributors
Mason Quicke
Winston Ivory
Ronan McDermott**

**Dataset Information**
Our project uses the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2020 dataset, specifically the cleaned version (heart_2020_cleaned.csv). The dataset contains approximately 300,000 survey responses with 18 features related to health status, lifestyle, and demographics.
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

**Setup Instructions**
Our project was developed and intended to be run in Google Colab or VSCode. 

**Dependencies / Libraries:** the libraries used were pandas, numpy, matplotlib, seaborn, scikit-learn**
Installation:** If running in Google Colab, these libraries are already pre-installed. If running locally, you can install them via terminal:
pip install pandas numpy matplotlib seaborn scikit-learn

**How to Run the Code**
For Collab: Download the heart_2020_cleaned.csv dataset and upload it to your Google Drive, or on your device.
Open the Jupyter Notebook in Google Colab.
Then mount the dataset, and make sure the model pulls it from where you stored it. From there run the model with the code pasted in. If there are any erros, it will most likely be the file isn't pulling the dataset from the right space.
Run the first cell to mount your Google Drive to the Colab environment.
In the second cell, update the file_path variable to point to the exact location of the dataset in your Google Drive.
Go to the top menu and select Runtime > Run all to execute the actual code.

For VSCode: 
Clone the repository and open the project folder in VS Code.
Ensure you have the Python and Jupyter extensions installed in VS Code.
Verify that heart_2020_cleaned.csv is located in the root directory alongside your code file.
Open the code notebook file, select your Python kernel environment, and click Run All at the top of the interface.
*The model optimization/hyperparameter tuning section utilizes cross-validation loops and may take a couple of minutes to complete execution.*

**Results & Methodology**
1. Data Cleaning & Preprocessing: Before building our models, we thoroughly cleaned and preprocessed the data. This involved checking for missing values, removing over 18,000 duplicate rows to prevent train/test leakage, binary encoding our target variable (HeartDisease), and one-hot encoding 13 categorical features. We also applied standard scaling to our numerical features to prepare them for algorithms like Logistic Regression and KNN.

2. The Naïve Baseline (The Accuracy issue we noticed): To establish the best point of comparison, we built a default, "naïve" baseline model using what is know as out-of-the-box algorithms with no class balancing or tuning. Because the dataset is highly imbalanced (only about 9% of respondents actually have heart disease), these naïve models suffered from the Accuracy appearing very high. For example, the naïve Logistic Regression model achieved a 91% accuracy simply by guessing "No Heart Disease" for almost everyone. However, its Recall was a terrible ~10%, meaning it missed 90% of actual heart disease cases, which can be a dangerous outcome for a clinical tool could save lives.

**3. Fully Improved Models:** 
To build a safe and effective screening models, we implemented several improvements:
Class Balancing: We used class_weight='balanced' for Logistic Regression and Decision Trees, and random undersampling for KNN, forcing the models to pay attention to the minority class.
Threshold Tuning: We carved out a completely isolated validation set to perform custom probability threshold tuning, shifting the decision threshold (from 0.50 at the start to 0.67 for Logistic Regression) in order to optimize for the F1 score and Recall. What we noticed was at a threshold of 0.67, the F1 was the best, but the recall dropped. In the context of our stakeholders, and us proposing this as a good tool for an initial screening, we believe preserving the strongest Recall was the most important, and at a threshold at 0.5, it performed better. So for an initial screening screening run 0.5, but anything past that the 0.67 threshold is better practice.

**4. Model Comparison & Final Selection:**
We evaluated three different algorithms: Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN).
Decision Tree: While highly interpretable and requiring no feature scaling, the optimized Decision Tree struggled slightly with generalization, achieving a final test Recall of ~49.5% and an F1 score of ~0.377. It was prone to slightly more overfitting.
K-Nearest Neighbors (KNN): KNN performed surprisingly well after balancing the data, achieving a Recall of ~53.4% and an F1 score of ~0.383. However, KNN is highly computationally expensive for predictions, making it impractical for deployment on a massive scale dataset like ours without significant down-sampling.
Why Logistic Regression is the Best: The optimized Logistic Regression model emerged as the clear winner. It achieved the highest Recall (55.5%), meaning it caught the largest percentage of actual heart disease cases. It also achieved the highest overall F1 score (0.391) and AUC-ROC (0.832), indicating the best balance of precision and sensitivity. Furthermore, Logistic Regression is computationally efficient and highly interpretable, allowing us to clearly see which features (like Age, Smoking, and General Health) drive the predictions via its coefficients.

**Acknowledgements & Resources**
Course Material: We relied heavily on our Zybooks course notes to help build the foundational machine learning code and understand the algorithms used in our project.
Google Colab Gemini: We utilized the built-in Google Colab Gemini AI feature to help explain, debug, and fix runtime errors as we were writing the code.
Claude (Anthropic): During the optimization phase, we struggled with preventing data leakage when tuning the decision thresholds for our improved models. We used Claude to help us understand and write the logic to correctly isolate a validation set for threshold tuning, ensuring our final test set remained completely unseen.
