# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset
data = pd.read_csv(r'C:\Users\Shehan Franciscu\Desktop\IW\completeSpamAssassin.csv', low_memory=False)

# Remove rows with invalid labels (not 0 or 1)
data = data[data['spam_or_not_spam'].isin(['0', '1'])]

# Drop rows with missing email content
data = data.dropna(subset=['email_content'])

# Data preprocessing
X = data['email_content']  # Features
y = data['spam_or_not_spam']  # Labels

# Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_rep)


#########GUI-Build##########

import tkinter as tk
from tkinter import messagebox, filedialog
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
model = joblib.load('model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize an empty list for email classifications
classifications = []

# Function to classify email and save a report
def classify_and_save_report():
    email_text = email_entry.get()
    if not email_text:
        messagebox.showinfo("Classification Result", "Please enter an email.")
        return

    # Preprocess the input email text
    email_text = clean_text(email_text)

    # Vectorize the email text
    email_tfidf = tfidf_vectorizer.transform([email_text])

    # Predict the email classification
    prediction = model.predict(email_tfidf)

    if prediction[0] == '0':
        result = "Not Spam"
    else:
        result = "Spam"

    classification_result_label.config(text=f"The email is classified as: {result}")

    # Append the email classification ('Spam' or 'Not Spam')
    classifications.append(result)

    # Save the email classification to a file
    report_filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])

    with open(report_filename, 'w') as report_file:
        report_file.write("Email Classification: {}\n".format(result))
        messagebox.showinfo("Report Saved", "Email classification has been saved.")

# Function for text cleaning and preprocessing
def clean_text(text):
    # Implement your text cleaning and preprocessing here
    # (Use the same preprocessing steps as when you trained the model)
    # Example: lowercase, remove punctuation, remove extra whitespaces, etc.
    return text

# Create the GUI window
window = tk.Tk()
window.title("Email Spam Classifier")

# Create and configure the input elements
email_label = tk.Label(window, text="Enter Email Text:")
email_label.pack()
email_entry = tk.Entry(window, width=50)
email_entry.pack()

classify_button = tk.Button(window, text="Classify and Save Report", command=classify_and_save_report)
classify_button.pack()

classification_result_label = tk.Label(window, text="")
classification_result_label.pack()

# Run the GUI
window.mainloop()