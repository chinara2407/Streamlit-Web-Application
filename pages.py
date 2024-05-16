import os
import xgboost
import catboost
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

title_image = Image.open("dsa.jpg")
page_image = Image.open("pages.jpg")
banner_image = Image.open("logo.jpeg")

st.set_page_config(
    page_title="Case Study",
    page_icon=banner_image,
    layout="wide"
)

st.title("Python Case Study 8")
st.text("Machine Learning Web Application with Streamlit")

st.sidebar.image(title_image)

page = st.sidebar.selectbox("", ["Homepage", "EDA", "Modeling"])

df_potability = pd.read_csv("water_potability.csv")
df_loan = pd.read_csv("loan_pred.csv")

if page == "Homepage":
    st.subheader("HOMEPAGE")
    st.image(page_image)

    data = st.selectbox("Select Dataset", ["Water Potability", "Loan Prediction"])

    st.write("Selected {} Dataset".format(data))

    st.warning("""
    Task: Our objective is to create an automation process.
    Pages should be assembled accordingly:
    1. The first page (i.e., homepage) should be used to introduce ourselves with the data.
    2. The second page, the EDA, should be used for data imputation, deletion, imbalance checking, cleaning data from outliers, and including some visuals for the data.
    3. The final page, the modeling page, should include preprocessing, a variety of scalers, a variety of encoders, train-test splitting distribution, building of the model, calculations for evaluation metrics, and some visualization.
    """)

    if data == "Water Potability":
        st.info("""
        PH – The PH Value of the Water;
                
        Hardness – Rigidity Value of the Water;
                
        Solids – The number of solids contained;
                
        Chloramines – The value of Chloramine components;
                
        Sulfate – The amount of Sulfate components;
                
        Conductivity – The value of conductivity specification;
                
        Organic_carbon – The amount of organic carbon components;
                
        Trihalomethanes – The amount of Trihalomethanes components;
        """)
    else:
        st.info("""
        Gender – Gender type;
                
        Married – Marital status;
                
        Dependents – Dependent variables;
                
        Education – Educational status;
                
        Self_Employed – The employment degree of the customer;
        
        ApplicantIncome – Income of the Applicant;
        
        CoapplicantIncome – Income of the Coapplicant;
        
        LoanAmount – The amount of Loan;
        
        Loan_Amount_Term – The term of Loan;
        
        Credit_History – Credit History;
        
        Property_Area – Area of Proportion;
        
        Loan_Status – Status of Loan;
        """)
elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    data = st.selectbox("Select Dataset", ["Water Potability Dataset", "Loan Prediction Dataset"])

    def outlier_treatment(column):
        sorted(column)
        Q1, Q3 = np.percentile(column, [25, 75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return lower, upper

    def describe(df):
        st.dataframe(df)
        st.subheader("Statistical Values")
        df.describe().T
        
        st.subheader("Balance of Data")
        st.bar_chart(df.iloc[:, -1].value_counts())

        column1, column2, column3 = st.columns([1,1,1])

        column1.subheader("NULL Variables")
        df_null = df.isnull().sum().to_frame().reset_index()
        df_null.columns = ["Columns", "Counts"]
        column1.dataframe(df_null)

        column2.subheader("Imputation")
        categorical_choice = column2.radio("Categorical", ["Mode", "Backfill", "Fill"])
        numerical_choice = column2.radio("Numerical", ["Mode", "Median"])

        column2.subheader("Feature Engineering")
        sampling = column2.checkbox("Under Sampling")
        outlier = column2.checkbox("Clean Outlier")

        if column2.button("Data Processing"):
            categorical_array = df.iloc[:, :-1].select_dtypes(include=["object"]).columns
            numerical_array = df.iloc[:, :-1].select_dtypes(exclude=["object"]).columns

            if len(categorical_array) > 0:
                if categorical_choice == "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[categorical_array] = imputer.fit_transform(df[categorical_array])
                elif categorical_choice == "Backfill":
                    df[categorical_array] = df[categorical_array].fillna(method='backfill')
                else:
                    df[categorical_array] = df[categorical_array].fillna(method='ffill')

            if len(numerical_array) > 0:
                if numerical_choice == "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='median')
                
                df[numerical_array] = imputer.fit_transform(df[numerical_array])

            column3.subheader("NON-NULL DataSet")
            null_df = df.isnull().sum().to_frame().reset_index()
            null_df.columns = ["Columns", "Counts"]
            column3.dataframe(null_df)

            if sampling:
                rus = RandomUnderSampler()
                X = df.iloc[:, :-1]
                y = df.iloc[:, [-1]]
                X,y=rus.fit_resample(X,y)
                df = pd.concat([X, y], axis=1)
                st.subheader("Balance of Data After UnderSampling")
                st.bar_chart(df.iloc[:, -1].value_counts())
            if outlier:
                for column in numerical_array:
                    lower, upper = outlier_treatment(df[column])
                    df[column] = np.clip(df[column], a_min=lower, a_max=upper)

            st.header("Processing finished successfully")

            st.subheader("Processed DataSet:")
            st.dataframe(df)

            st.subheader("Correlation HeatMap")
            heatmap = px.imshow(df[numerical_array].corr(), text_auto=True)
            st.plotly_chart(heatmap)

            if os.path.exists("formodel.csv"):
                os.remove("formodel.csv")
            df.to_csv("formodel.csv", index=False)

    if data == "Water Potability Dataset":
        describe(df_potability)
    else:
        describe(df_loan)
else:
    st.header("Modelling")
    if not os.path.exists("formodel.csv"):
        st.header("Please Run Preprocessing!")
    else:
        df = pd.read_csv("formodel.csv")
        st.dataframe(df)

        column1, column2 = st.columns([1, 1])
        scale_method = column1.radio("Scalers", ["Standard", "Robust", "MinMax"])
        encode_method = column2.radio("Encoders", ["Label", "OneHot"])

        st.header("Train and Test Splitting")
        col1, col2, col3 = st.columns([1, 1, 1])

        random_state = col1.text_input("Random State")
        split_size = col2.text_input("Split Size")
        model = col3.selectbox("Select Model", ["XGBoost", "CatBoost"])

        st.markdown("Selected {} model".format(model))

        if st.button("Run Model"):
            categorical_array = df.iloc[:,:-1].select_dtypes(include=["object"]).columns
            numerical_array = df.iloc[:,:-1].select_dtypes(exclude=["object"]).columns

            if len(numerical_array) > 0:
                if scale_method == "Standard":
                    scaler = StandardScaler()
                elif scale_method == "Robust":
                    scaler = RobustScaler()
                else:
                    scaler = MinMaxScaler()
                df[numerical_array] = scaler.fit_transform(df[numerical_array])

            if len(categorical_array) > 0:
                if encode_method == "Label":
                    encoder = LabelEncoder()
                    for column in categorical_array:
                        df[column] = encoder.fit_transform(df[column])
                else:
                    dfd=pd.get_dummies(df[categorical_array], drop_first=True)
                    df_=df.drop(df[categorical_array], axis=1)
                    df = pd.concat([df_,dfd], axis=1)
            st.dataframe(df)
            X=df.drop(df.iloc[:, [-1]], axis=1)
            y=df.iloc[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(split_size), random_state=int(random_state))


            st.markdown["X_train Size: {}".format(X_train.shape)]
            st.markdown["X_test Size: {}".format(X_test.shape)]
            st.markdown["Y_train Size: {}".format(y_train.shape)]
            st.markdown["Y_test Size: {}".format(y_test.shape)]
            
            
            
            if model == "XGBoost":
                booster = xgboost.XGBClassifier()
                #booster.fit(X_train,y_train)
            else:
                 booster = catboost.CatBoostClassifier()
            #     # booster.fit(X_train,y_train)

            booster.fit(X_train, y_train)
            y_pred = booster.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            st.subheader("Your Accuracy Score is: {}".format(np.round(score)))

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report)
            st.dataframe(df_report)

            st.subheader("Confusion Matrix")
            st.dataframe(confusion_matrix(y_test, y_pred))

            fpr, tpr, thresholds = roc_curve(y_test, booster.predict_proba(X_test)[:, -1])
            fig = px.area(x=fpr, y=tpr, title="ROC Curve", labels={"x":"False Positive Area", "Y":"True Positive Rate"})
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

            st.plotly_chart(fig)
            st.markdown("AUC Score: {}".format(roc_auc_score(y_test, y_pred)))
            st.title("Thanks For Using!")
