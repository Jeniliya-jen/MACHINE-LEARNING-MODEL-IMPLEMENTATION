# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: JENILIYA J

*INTERN ID*: CT04DG574

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*:  NEELA SANTHOSH

# MACHINE-LEARNING-MODEL-IMPLEMENTATION - CodTech Internship Task 4:
This project is part of the CodTech Python Internship Program and focuses on implementing a machine learning model using Scikit-learn to detect spam messages. With the rapid digitalization of communication, especially through emails and SMS, spam detection has become increasingly important. The primary goal of this task is to build a predictive model capable of classifying a given message as “spam” or “ham” (not spam) using a dataset and present it professionally in a Jupyter Notebook.

## Project Overview:
This project is a beginner-friendly implementation of a **Spam Email Detection model** using **scikit-learn** in a **Jupyter Notebook**. The aim is to classify messages as either ham (not spam) or spam, based on the content of the email. This is a common real-world application of machine learning and natural language processing (NLP) used in email services.

The solution involves transforming raw text data into a numerical format that a machine learning algorithm can understand, training the model, and evaluating how accurately it performs on unseen data.

The objective of this project is to develop a supervised learning model that can accurately classify messages into two categories: spam and ham. To achieve this, a machine learning pipeline is constructed using tools like Pandas for **data manipulation, Scikit-learn for preprocessing** and **model training**, and **Matplotlib/Seaborn for result visualization**. The model is trained on a labeled dataset consisting of example messages, each categorized as spam or ham. The model should be able to learn from patterns within the data and apply this knowledge to classify unseen messages effectively.

## Dataset Overview:
The dataset used is a CSV file named spam_email_data.csv, which contains 90 rows of labeled SMS/email messages. Each row has two columns:
- **Label**: Indicates whether the message is ham or spam.
- **Message**: Contains the actual text of the message.
The dataset is manually curated with a healthy mix of real-sounding spam and ham messages. This ensures a balanced learning experience and helps avoid overfitting. The labels are converted from textual format (‘ham’ and ‘spam’) to binary (0 and 1) to aid in machine learning model training.

## Tools and Libraries Used:
The project uses the following Python libraries:
- **pandas**: For reading and manipulating the dataset.
- **scikit-learn**: For model training, feature extraction, and performance evaluation.
- **matplotlib** and **seaborn**: For visualizing the confusion matrix.
- **CountVectorizer**: For text preprocessing
- **Naive Bayes Classifier (MultinomialNB)**: As the predictive algorithm
- **Jupyter Notebook**: As the development environment.

## Model Workflow:
1. **Data Loading**: The CSV file is read into a Pandas DataFrame.
2. **Label Encoding**: Labels are converted from ham and spam into binary format using mapping (ham → 0, spam → 1).
3. **Text Vectorization**: The text messages are converted into numerical format using Bag-of-Words via CountVectorizer, making them suitable for machine learning algorithms.
4. **Data Splitting**: The data is split into a training set and a test set (80:20 ratio).
5. **Model Training**: A MultinomialNB (Naive Bayes) classifier is trained on the vectorized messages.
6. **Prediction**: The trained model is used to make predictions on the test data.
7. **Evaluation**: The model is evaluated using Accuracy Score, Classification Report (precision, recall, F1-score), and a Confusion Matrix.
8. **Sample Output**: A few predictions from the test set are displayed, showing the model’s understanding of spam vs ham messages.

## What it does:
1.**Model Implementation**:
- Loaded the dataset using pandas
- Preprocessed the text data using CountVectorizer
- Split the data into training and testing sets
- Trained a Naive Bayes classifier using MultinomialNB
- Made predictions on test data

2.**Model Evaluation**:
- Printed accuracy score
- Displayed a classification report (precision, recall, F1-score)
- Visualized the confusion matrix using Seaborn/Matplotlib
- Shown sample predictions to demonstrate how the model performs on unseen text

## Results:
The model achieved an accuracy of 83.33% on the test dataset, which indicates reliable performance for a simple spam detection task. The classification report provides deeper insights into how well the model performs on both spam and ham messages, and the confusion matrix helps visualize the number of true positives, false positives, etc. The sample predictions section showcases five examples from the test set along with the predicted label, demonstrating the model’s effectiveness.

To illustrate the effectiveness of the spam classification model, a small portion of the test dataset was used to generate sample predictions. These messages may appear **jumbled** or **out of grammatical order**. This is because they were reconstructed using the **inverse_transform() method** after vectorization. While the full message structure is not restored, the words shown are meaningful indicators used by the model to determine whether the message is spam or not.

## Conclusion:
This project successfully demonstrates the implementation of a **Spam Email Detection model** using **Machine Learning techniques**. By leveraging the **Naive Bayes classification algorithm** and **CountVectorizer for text preprocessing**, the model was able to classify email messages as either **spam or ham (non-spam)** with commendable accuracy. The use of **Jupyter Notebook** allowed for an organized, interactive, and explainable workflow, from loading and preprocessing the dataset to training the model and evaluating its performance using accuracy metrics and a confusion matrix.

This task provided valuable hands-on experience with the complete ML pipeline — including **data cleaning, vectorization, model training, evaluation, and visualization**. It showcased how real-world problems like spam detection can be effectively tackled using simple yet powerful tools like scikit-learn.

I would like to extend my sincere thanks to **CodTech It Solutions** for providing this internship opportunity and for assigning such a practical and insightful task. It enabled me to apply theoretical knowledge in a real-world context and strengthened my foundation in machine learning, particularly in text classification. I am grateful for the learning experience.

## Output:
![Image](https://github.com/user-attachments/assets/9c6a3a57-9b60-4aa7-8ac9-64778c55c747)


