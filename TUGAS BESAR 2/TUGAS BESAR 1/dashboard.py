
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm

# Load Model Klasifikasi
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

# Fungsi Prediksi Klasifikasi
def prediction(Gender, Age, Academic_Pressure, Study_Satisfaction, Sleep_Duration,
               Dietary_Habits, Have_you_ever_had_suicidal_thoughts, Study_Hours,
               Financial_Stress, Family_History_of_Mental_Illness):
    input_data = np.array([[Gender, Age, Academic_Pressure, Study_Satisfaction, Sleep_Duration,
                            Dietary_Habits, Have_you_ever_had_suicidal_thoughts, Study_Hours,
                            Financial_Stress, Family_History_of_Mental_Illness]])
    pred = classifier.predict(input_data)
    proba = classifier.predict_proba(input_data)[:, 1]
    return pred, proba


# Fungsi Evaluasi Model
def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba


# Fungsi Plot ROC Curve
def plot_roc_curve(fpr, tpr, auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    st.pyplot(plt)


# Fungsi Plot Confusion Matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"], yticklabels=["Actual No", "Actual Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)


# Fungsi Plot Perbandingan Akurasi
def plot_accuracy_comparison(accuracy, accuracy_best):
    plt.figure(figsize=(10, 8))
    bars = plt.bar(['Before Tuning', 'After Tuning'], [accuracy, accuracy_best], color=['green', 'blue'])
    plt.ylim(0, 1)
    plt.ylabel('Akurasi')
    plt.title('Perbandingan Akurasi Model Regresi Logistik\nSebelum dan Sesudah Tuning')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, f'{yval:.2f}', ha='center', va='center', color='white', fontweight='bold')
    st.pyplot(plt)


# Fungsi Utama
def main():
    st.title("Analisis Data Tingkat Stress Akademik Mahasiswa")
    st.sidebar.title("Pilih Metode Analisis")
    option = st.sidebar.selectbox("Metode:", ["Klasifikasi", "Regresi Logistik"])

    if option == "Klasifikasi":
        st.subheader("Analisis Klasifikasi")
        df = pd.read_csv("Depression Student Dataset.csv")

        # Data Preprocessing
        cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']
        for i in cols:
            df[i] = df[i].apply(lambda x: 1 if x == 'Yes' else 0)
        df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'Female' else 1)
        df['Sleep Duration'] = df['Sleep Duration'].map({'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4})
        df['Dietary Habits'] = df['Dietary Habits'].map({'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3})

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(df.drop(columns=['Depression']), df['Depression'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        accuracy, precision, recall, f1, roc_auc, y_pred, y_pred_proba = evaluate_model(X_test, y_test, classifier)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success(f"Accuracy: {accuracy:.2f}")
        with col2:
            st.info(f"Precision: {precision:.2f}")
        with col3:
            st.warning(f"Recall: {recall:.2f}")
        with col4:
            st.error(f"F1 Score: {f1:.2f}")

        plot_option = st.selectbox("Select Plot:", ["Select", "ROC Curve", "Confusion Matrix"])
        if plot_option == "ROC Curve":
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plot_roc_curve(fpr, tpr, roc_auc)
        elif plot_option == "Confusion Matrix":
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm)

        Gender = st.text_input("Gender")
        Age = st.text_input("Age")
        Academic_Pressure = st.text_input("Academic Pressure")
        Study_Satisfaction = st.text_input("Study Satisfaction")
        Sleep_Duration = st.text_input("Sleep Duration")
        Dietary_Habits = st.text_input("Dietary Habits")
        Have_you_ever_had_suicidal_thoughts = st.text_input("Have you ever had suicidal thoughts ?")
        Study_Hours = st.text_input("Study Hours")
        Financial_Stress = st.text_input("Financial Stress")
        Family_History_of_Mental_Illness = st.text_input("Family History of Mental Illness")

        result = ""
        proba_result = ""

        if st.button("Predict"):
            result, proba = prediction(Gender, Age, Academic_Pressure, Study_Satisfaction, Sleep_Duration, Dietary_Habits, Have_you_ever_had_suicidal_thoughts, Study_Hours, Financial_Stress, Family_History_of_Mental_Illness)
            proba_result = f"{proba[0]:.2f}"
            result = "Yes" if result[0] == 1 else "No"

        st.success(f"Prediksi Depresi: {result}")
        st.success(f"Probabilitas Depresi: {proba_result}")

    elif option == "Regresi Logistik":
        st.subheader("Analisis Regresi Logistik")
        df = pd.read_csv("Depression Student Dataset.csv")

        # Preprocessing
        binary_mappings = {'Gender': {'Male': 1, 'Female': 0},
                           'Have you ever had suicidal thoughts ?': {'Yes': 1, 'No': 0},
                           'Family History of Mental Illness': {'Yes': 1, 'No': 0},
                           'Depression': {'Yes': 1, 'No': 0}}
        df.replace(binary_mappings, inplace=True)

        df['Sleep Duration'] = df['Sleep Duration'].map({'Less than 5 hours': 1, '5-6 hours': 2, '7-8 hours': 3, 'More than 8 hours': 4})
        df['Dietary Habits'] = df['Dietary Habits'].map({'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3})

        X = df.drop(columns=['Depression'])
        y = df['Depression']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
        grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(X_test)
        accuracy_best = accuracy_score(y_test, y_pred_best)

        col1, col2 = st.columns(2)
        with col1:
            st.error(f"Accuracy Before Tuning: {accuracy:.2f}")
        with col2:
            st.success(f"Accuracy After Tuning: {accuracy_best:.2f}")

        plot_option = st.selectbox("Select Plot:", ["Select", "Confusion Matrix Before Tuning", "Confusion Matrix After Tuning", "Accuracy Comparison"])
        if plot_option == "Confusion Matrix Before Tuning":
            plot_confusion_matrix(confusion_matrix(y_test, y_pred))
        elif plot_option == "Confusion Matrix After Tuning":
            plot_confusion_matrix(confusion_matrix(y_test, y_pred_best))
        elif plot_option == "Accuracy Comparison":
            plot_accuracy_comparison(accuracy, accuracy_best)

if __name__ == '__main__':
    main()
