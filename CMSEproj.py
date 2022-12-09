import streamlit as st
st. set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import altair as alt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.decomposition import PCA

#--------------------------------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
my_page = st.sidebar.radio('Page Navigation', ['EDA', 'ML'])

if my_page == 'ML':
    st.title('ML')
    #Titles
    tit1,tit2 = st.columns((4, 1))
    tit1.markdown("<h1 style='text-align: center;'><u>Heart Disease Predictions</u> </h1>",unsafe_allow_html=True)
    # tit2.image("healthcare2.png")
    st.sidebar.title("Dataset and Classifier")

    dataset_name=st.sidebar.selectbox("Select Dataset: ",('Heart Attack',"Breast Cancer"))
    classifier_name = st.sidebar.selectbox("Select Classifier: ",("Logistic Regression","KNN","SVM","Decision Trees",
                                                                "Random Forest","Gradient Boosting","XGBoost"))

    LE=LabelEncoder()
    def get_dataset(dataset_name):
        if dataset_name=="Heart Attack":
            data=pd.read_csv("https://raw.githubusercontent.com/souradipmsu/final-project/main/heartproj.txt")
            st.header("Heart Attack Prediction")
            return data

        else:
            data=pd.read_csv("https://raw.githubusercontent.com/souradipmsu/final-project/main/BreastCancer.txt")
            
            data["diagnosis"] = LE.fit_transform(data["diagnosis"])
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
            st.header("Breast Cancer Prediction")
            return data

    data = get_dataset(dataset_name)

    def selected_dataset(dataset_name):
        if dataset_name == "Heart Attack":
            X=data.drop(["output"],axis=1)
            Y=data.output
            return X,Y

        elif dataset_name == "Breast Cancer":
            X = data.drop(["id","diagnosis"], axis=1)
            Y = data.diagnosis
            return X,Y

    X,Y=selected_dataset(dataset_name)

    #Plot output variable
    def plot_op(dataset_name):
        col1, col2 = st.beta_columns((1, 5))
        plt.figure(figsize=(12, 3))
        plt.title("Classes in 'Y'")
        if dataset_name == "Heart Attack":
            col1.write(Y)
            sns.countplot(Y, palette='gist_heat')
            col2.pyplot()

        elif dataset_name == "Breast Cancer":
            col1.write(Y)
            sns.countplot(Y, palette='gist_heat')
            col2.pyplot()

    st.write(data)
    # st.write("Shape of dataset: ",data.shape)
    # st.write("Number of classes: ",Y.nunique())
    plot_op(dataset_name)


    def add_parameter_ui(clf_name):
        params={}
        st.sidebar.write("Select values: ")

        if clf_name == "Logistic Regression":
            R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
            MI = st.sidebar.slider("max_iter",50,400,step=50)
            params["R"] = R
            params["MI"] = MI

        elif clf_name == "KNN":
            K = st.sidebar.slider("n_neighbors",1,20)
            params["K"] = K

        elif clf_name == "SVM":
            C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
            kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
            params["C"] = C
            params["kernel"] = kernel

        elif clf_name == "Decision Trees":
            M = st.sidebar.slider("max_depth", 2, 20)
            C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
            SS = st.sidebar.slider("min_samples_split",1,10)
            params["M"] = M
            params["C"] = C
            params["SS"] = SS

        elif clf_name == "Random Forest":
            N = st.sidebar.slider("n_estimators",50,500,step=50,value=100)
            M = st.sidebar.slider("max_depth",2,20)
            C = st.sidebar.selectbox("Criterion",("gini","entropy"))
            params["N"] = N
            params["M"] = M
            params["C"] = C

        elif clf_name == "Gradient Boosting":
            N = st.sidebar.slider("n_estimators", 50, 500, step=50,value=100)
            LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
            L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
            M = st.sidebar.slider("max_depth",2,20)
            params["N"] = N
            params["LR"] = LR
            params["L"] = L
            params["M"] = M

        elif clf_name == "XGBoost":
            N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
            LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
            O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
            M = st.sidebar.slider("max_depth", 1, 20,value=6)
            G = st.sidebar.slider("Gamma",0,10,value=5)
            L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
            A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
            CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
            params["N"] = N
            params["LR"] = LR
            params["O"] = O
            params["M"] = M
            params["G"] = G
            params["L"] = L
            params["A"] = A
            params["CS"] = CS

        RS=st.sidebar.slider("Random State",0,100)
        params["RS"] = RS
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name,params):
        global clf
        if clf_name == "Logistic Regression":
            clf = LogisticRegression(C=params["R"],max_iter=params["MI"])

        elif clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])

        elif clf_name == "SVM":
            clf = SVC(kernel=params["kernel"],C=params["C"])

        elif clf_name == "Decision Trees":
            clf = DecisionTreeClassifier(max_depth=params["M"],criterion=params["C"],min_impurity_split=params["SS"])

        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"],criterion=params["C"])

        elif clf_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(n_estimators=params["N"],learning_rate=params["LR"],loss=params["L"],max_depth=params["M"])

        # elif clf_name == "XGBoost":
            # clf = XGBClassifier(booster="gbtree",n_estimators=params["N"],max_depth=params["M"],learning_rate=params["LR"],
                                # objective=params["O"],gamma=params["G"],reg_alpha=params["A"],reg_lambda=params["L"],colsample_bytree=params["CS"])

        return clf

    clf = get_classifier(classifier_name,params)

    #Build Model
    def model():
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65)

        #MinMax Scaling / Normalization of data
        Std_scaler = StandardScaler()
        X_train = Std_scaler.fit_transform(X_train)
        X_test = Std_scaler.transform(X_test)

        clf.fit(X_train,Y_train)
        Y_pred = clf.predict(X_test)
        acc=accuracy_score(Y_test,Y_pred)

        return Y_pred,Y_test

    Y_pred,Y_test=model()

    #Plot Output
    def compute(Y_pred,Y_test):
        #Plot PCA
        pca=PCA(2)
        X_projected = pca.fit_transform(X)
        x1 = X_projected[:,0]
        x2 = X_projected[:,1]
        plt.figure(figsize=(16,8))
        plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()
        st.pyplot()

        c1, c2 = st.columns((4,3))
        #Output plot
        plt.figure(figsize=(12,6))
        plt.scatter(range(len(Y_pred)),Y_pred,color="yellow",lw=5,label="Predictions")
        plt.scatter(range(len(Y_test)),Y_test,color="red",label="Actual")
        plt.title("Prediction Values vs Real Values")
        plt.legend()
        plt.grid(True)
        c1.pyplot()

        #Confusion Matrix
        cm=confusion_matrix(Y_test,Y_pred)
        class_label = ["High-risk", "Low-risk"]
        df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
        plt.figure(figsize=(12, 7.5))
        sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
        plt.title("Confusion Matrix",fontsize=15)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        c2.pyplot()

        #Calculate Metrics
        acc=accuracy_score(Y_test,Y_pred)
        mse=mean_squared_error(Y_test,Y_pred)
        precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
        st.subheader("Metrics of the model: ")
        st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
            round(precision, 3), round(recall, 3), round(fscore,3), round((acc*100),3), round((mse),3)))

    st.markdown("<hr>",unsafe_allow_html=True)
    st.header(f"1) Model for Prediction of {dataset_name}")
    st.subheader(f"Classifier Used: {classifier_name}")
    compute(Y_pred,Y_test)

    #Execution Time
    end_time=time.time()
    # st.info(f"Total execution time: {round((end_time - start_time),4)} seconds")

    #Get user values
    def user_inputs_ui(dataset_name,data):
        user_val = {}
        if dataset_name == "Breast Cancer":
            X = data.drop(["id","diagnosis"], axis=1)
            for col in X.columns:
                name=col
                col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
                user_val[name] = round((col),4)

        elif dataset_name == "Heart Attack":
            X = data.drop(["output"], axis=1)
            for col in X.columns:
                name=col
                col = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
                user_val[name] = col

        return user_val

    #User values
    st.markdown("<hr>",unsafe_allow_html=True)
    st.header("2) User Values")
    with st.expander("See more"):
        st.markdown("""
        In this section you can use your own values to predict the target variable. 
        Input the required values below and you will get your status based on the values. <br>
        <p style='color: red;'> 1 - High Risk </p> <p style='color: green;'> 0 - Low Risk </p>
        """,unsafe_allow_html=True)

    user_val=user_inputs_ui(dataset_name,data)

    #@st.cache(suppress_st_warning=True)
    def user_predict():
        global U_pred
        if dataset_name == "Breast Cancer":
            X = data.drop(["id","diagnosis"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])

        elif dataset_name == "Heart Attack":
            X = data.drop(["output"], axis=1)
            U_pred = clf.predict([[user_val[col] for col in X.columns]])

        st.subheader("Your Status: ")
        if U_pred == 0:
            st.write(U_pred[0], " - You are not at high risk :)")
        else:
            st.write(U_pred[0], " - You are at high risk :(")
    user_predict()  #Predict the status of user.
        

else:
    st.title('EDA')
    st.markdown("<h1 style='text-align: center;'><u>Heart Disease Predictions</u> </h1>",unsafe_allow_html=True)
    heart_dataset = pd.read_csv("https://raw.githubusercontent.com/souradipmsu/final-project/main/heart.csv")

    alt_handle=alt.Chart(heart_dataset).mark_circle(size=60).encode(
        x='age',
        y='thalach',
        color='sex',
        tooltip=['age', 'sex', 'chol', 'thalach']
    ).interactive()
    st.altair_chart(alt_handle)

    alt_handle = alt.Chart(heart_dataset).mark_circle(size=60).encode(x='sex', y='age').interactive()
    st.altair_chart(alt_handle)
    st.write(""" #In the above graph, it can be said that in this dataset both males and females have
    equal chances of heart disease. But one observation which is unique from this dataset is that in the 
    age group of 70-80 for females, the chances of detection of heart diseases is low. """)


    alt_handle=alt.Chart(heart_dataset).mark_point().encode(x="chol", y="sex").interactive()
    st.altair_chart(alt_handle)

    st.write(""" #In the above visualization, we can notice that males have usually high cholestrol levels 
    as compared to females.As we can see from this dataset, there are some males whose cholestrol levels 
    have exceeded 400 whereas for females, the maximum is around 350. """)


    # alt_handle=alt.Chart(heart_dataset).mark_point().encode(x="restecg",y="slope").interactive()
    # st.altair_chart(alt_handle)

    alt_handle=alt.Chart(heart_dataset).mark_boxplot(extent='min-max').encode(
        x='age:O',
        y='fbs:Q'
    )
    st.altair_chart(alt_handle)
    st.write(""" #In the above visualization, the condition is that if the fasting blood sugar is equal to
    or above 120mg/dl,the fbs is indicated as 1 in the graph.So here we can see that apart from the age group 
    range of 55-65, mostly patients have fasting blood sugar greater than 120mg/dl which 
    significantly increases their chances of heart diseases. """)


    alt_handle=alt.Chart(heart_dataset).transform_density(
        'chol',
        as_=['chol', 'restecg'],
        extent=[5, 50],
        groupby=['Sex']
    ).mark_area(orient='horizontal').encode(
        y='chol:Q',
        color='sex:N',
        x=alt.X(
            'exang:Q',
            stack='center',
            impute=None,
            title=None,
            axis=alt.Axis(labels=False, values=[0],grid=False, ticks=True),
        ),
        column=alt.Column(
            'ex:N',
            header=alt.Header(
                titleOrient='bottom',
                labelOrient='bottom',
                labelPadding=0,
            ),
        )
    ).properties(
        width=100
    ).configure_facet(
        spacing=0
    ).configure_view(
        stroke=None
    )
    st.altair_chart(alt_handle)

    alt_handle=alt.Chart(heart_dataset).mark_circle(size=60).encode(
        x='chol',
        y='thalach',
        color='sex',
        tooltip=['age', 'sex', 'chol', 'thalach']
    ).interactive()
    st.altair_chart(alt_handle)



    alt.Chart(heart_dataset).mark_bar().encode(
        x='chol:O',
        y="trestbos:Q",
        # The highlight will be set on the result of a conditional statement
        color=alt.condition(
            alt.datum.year == 1810,  # If the year is 1810 this test returns True,
            alt.value('orange'),     # which sets the bar orange.
            alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
        )
    ).properties(width=600)

    st.write(""" #In the above visualization, we can notice that the maximum heart rate achieved is inversely
    proportional to the age of the patient. With increasing age, heart rate decreases. It can also be 
    inferred that in terms of heart rate, there are no significant differences between men and women.
     """)
    measurements = ['trestbps', 'chol', 'thalach']
    # categorical features
    categories = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
    st.sidebar.markdown("### Blood Measurements (Scatter Plot)")

    x_axis = st.sidebar.selectbox("X-Axis", measurements)
    y_axis = st.sidebar.selectbox("Y-Axis", measurements, index=1)
    if x_axis and y_axis:
        scatter_fig = plt.figure(figsize=(12,8))
        scatter_ax = scatter_fig.add_subplot(111)
        low_attack = heart_dataset[heart_dataset["target"] == 0]
        high_attack = heart_dataset[heart_dataset["target"] == 1]
        low_attack.plot.scatter(x=x_axis, y=y_axis, s=120, c="tomato", alpha=0.6, ax=scatter_ax, label="High risk of heart attack")
        high_attack.plot.scatter(x=x_axis, y=y_axis, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax,
                           title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Low risk of heart attack");





        




