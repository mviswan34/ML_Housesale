from warnings import catch_warnings
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import pydeck as pdk
import altair as alt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

st.sidebar.title('Welcome to Ames House Sale Project')
page = st.sidebar.radio(
    "What would you like to know about this project ?",
    ("EDA", 'Prediction Model')
)


if page == "EDA":
    df = pd.read_csv("Real_Estate.csv")
    st.header('Housing prices in Ames, Iowa')
    col1, col2 = st.columns(2)
    img1 = "https://www.zip-area.com/iowa/images/map_ames.png"
    col1.image(img1, use_column_width=True)
    img2 = "https://www.fau.edu/newsdesk/images/news/overpriced-homes.jpg"
    col2.image(img2, use_column_width=True)

    nbhd_dict1 = {'Blmngtn':'Bloomington Heights','Blueste':'Bluestem','BrDale':'Briardale', 'BrkSide':'Brookside', 'ClearCr':'Clear Creek', 'CollgCr':'College Creek', 'Crawfor':'Crawford', 'Edwards':'Edwards', 'Gilbert':'Gilbert', 'IDOTRR':'Iowa DOT and Rail Road', 'MeadowV':'Meadow Village', 'Mitchel':'Mitchell', 'NAmes':'North Ames', 'NoRidge':'Northridge', 'NPkVill':'Northpark Villa', 'NridgHt':'Northridge Heights', 'NWAmes':'Northwest Ames', 'OldTown':'Old Town', 'SWISU':'South & West of Iowa State University', 'Sawyer':'Sawyer', 'SawyerW':'Sawyer West', 'Somerst':'Somerset', 'StoneBr':'Stone Brook', 'Timber':'Timberland', 'Veenker':'Veenker'}
    nbhd_dict2 = {'Bloomington Heights':'Blmngtn','Bluestem':'Blueste','Briardale':'BrDale', 'Brookside':'BrkSide', 'Clear Creek':'ClearCr', 'College Creek':'CollgCr', 'Crawford':'Crawfor', 'Edwards':'Edwards', 'Gilbert':'Gilbert', 'Iowa DOT and Rail Road':'IDOTRR', 'Meadow Village':'MeadowV', 'Mitchell':'Mitchel', 'North Ames':'Names', 'Northridge':'NoRidge', 'Northpark Villa':'NPkVill', 'Northridge Heights':'NridgHt', 'Northwest Ames':'NWAmes', 'Old Town':'OldTown', 'South & West of Iowa State University':'SWISU', 'Sawyer':'Sawyer', 'Sawyer West':'SawyerW', 'Somerset':'Somerst', 'Stone Brook':'StoneBr', 'Timberland':'Timber', 'Veenker':'Veenker'}
    ## Introduction
    st.markdown("#### Introduction")
    st.markdown("The small American city of Ames is situated in central Iowa's Story County. Ames, Iowa is a little town with a heart that beats to the rhythm of a much bigger city. There are no more than 65,000 people living here.")
    height = 250
    dis_map = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [42.0345, -93.6203],
        columns=['lat', 'lon'])
    st.map(dis_map, zoom=None, use_container_width=True)
    st.markdown("In this growing city, the housing and real estate markets are also evolving quickly.The idea behind this application is to help users understand the housing market and provide them with information on the sale prices of the properties they are considering.")
    ### About the dataset
    st.markdown("#### Data")
    st.markdown("##### Story behind the Data :")
    st.markdown("This data set was constructed by Professor Dean De Cock from truman State University for the purpose of an end of semester project for an undergraduate regression course. The original data was obtained directly from the Ames Assessor's Office and it was used for tax assessment purposes. But it lends itself directly to the prediction of home selling prices. The type of information contained in the data was similar to what a typical home buyer would want to know before making a purchase. ")
    st.markdown("##### About Dataset")
    st.markdown("The dataset contains 81 features describing a wide range of characteristics of 1,460 homes in the city of Ames.")
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(150, 50, 49);
    }
    </style>""", unsafe_allow_html=True)
    if st.button(label = "Click to load a sample of the dataset"):
        st.write(df.head(5))
    st.write("See [here](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt) for detailed description of all the columns.")
    st.markdown("#### Goals")
    st.markdown("- To help make the real estate investment process easier for prospective buyers. Ideally, it takes a client to visit a 10 places in a span of 2 weeks to focus on one particular property that meets most of their interest. The analysis on this application helps them to narrow down the list of properties based on their requirements.")
    st.markdown("- Find the sale price of the properties. Sale price is affected by a number of features such as the property size, geographical appeal, facilities close by, the number of rooms, the type of building materials used, and the age and condition of the building")
    st.markdown("- From a data science perspective, this application provides an excellent example for EDA of a dataset.")
    #Beginning of the application
    st.subheader("Let's begin !!!")
    ## Selecting the type of Property
    st.markdown("##### What kind of real estate are you seeking? Please Choose: ")
    cb1 = st.checkbox("Residential")
    cb2 = st.checkbox("Industrial")
    cb3 = st.checkbox("Agricultural")
    cb4 = st.checkbox("Commercial")
    type = []
    type_select = False
    if cb1:
        type = type + ['FV','RH','RL','RP','RM']
        type_select = True
    if cb2:
        type_select = True
        type = type + ['I']
    if cb3:
        type_select = True
        type = type + ['A']
    if cb4:
        type_select = True
        type = type + ['C']
    if type_select:
        df_mod = df.loc[df['MSZoning'].isin(type)]
    else:
        df_mod = df
    st.markdown("Depending on your selection, the plot below displays the number of properties for sale in each of Ames' neighborhoods.")
    st.write("NOTE: No option has been choosen. The plotted graph shows every type of property listed in Ames.")
    bars = alt.Chart(df_mod).mark_bar().encode(
        alt.Y('Neighborhood:N'),
        alt.X('count(Neighborhood):Q'),)
    text = bars.mark_text(
        align='left',
        baseline='middle',
        color = 'Orange',
        dx=2  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(text='count(Neighborhood)')
    fig1 = (bars + text).properties(height=400)
    st.altair_chart(fig1, use_container_width=True)
    ## Neighborhood Wise Selection.
    df_mod2 = df_mod
    nbhood = df_mod2.Neighborhood.unique()
    nbhd = np.array([nbhd_dict1[x] for x in nbhood])    
    st.markdown("##### Would you be interested in learning how the properties in each neighborhood compare against one another?")
    st.write("If so, please choose your preferred Neighborhood below to learn how the houses listed here compare in terms of Sale Price and general quality.")
    nb = st.multiselect(label="Please select one or more neighborhoods:", options=nbhd, default = None)
    nbhood2 = [nbhd_dict2[x] for x in nb]
    nb_select = False
    if len(nb)!=0:
        df_mod2 = df_mod2.loc[df_mod2['Neighborhood'].isin(nbhood2)]
        nb_select = True
    st.write("The distribution of properties in the chosen neighborhood according to their general state is shown in the plot below.")
    st.write("NOTE: Your previous selection of the property type is also reflected.")
    brush = alt.selection(type='interval')
    points = alt.Chart(df_mod2).mark_point().encode(
        x='OverallQual:Q',
        y='SalePrice:Q',
        color=alt.condition(brush, 'Neighborhood:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )
    bars = alt.Chart(df_mod2).mark_bar().encode(
        y='Neighborhood:N',
        color='Neighborhood:N',
        x='count(Neighborhood):Q'
    ).transform_filter(
        brush
    )
    if len(nb)!=0:
        fig2 = (points & bars)  
    else:
        fig2 = (points).properties(height=400)
    st.altair_chart(fig2, use_container_width=True)
    st.write("We can determine the sale price from the above mentioned plot according to the overall quality of the listed properties. The sale price typically rises in line with the improvement in property quality.")
    #Neighborhood Heatmap
    st.markdown("##### Not sure about the Neighborhood selection? Click on the following button:")
    if st.button(label="Click Me"):
        st.write("The below histplot may help you choose one. The density makes it easier to identify regions where there are the most properties available, signaling that more people will come in like you!")
        if len(nb)!=0:
            st.write("NOTE: The heatmap displays only for the neighborhoods you have selected above.")
        fig3 = alt.Chart(df_mod2).mark_rect().encode(
            x='Neighborhood:O',
            y='OverallQual:O',
            color='count():Q'
        )
        st.altair_chart(fig3, use_container_width=True)
        st.write("Now that you are aware of the areas where the high-quality homes are located, you can select one or more of the neighborhood options listed above.")
    ## House Size Selection
    st.markdown("##### The size of a property is a significant element in determining its price.")
    df_mod3=df_mod2
    ar = st.slider(label="Please indicate the minimum square footage you desire in a property:", min_value = int(df_mod3['GrLivArea'].min()), max_value=int(df_mod3['GrLivArea'].max()))
    df_mod3 = df_mod3.loc[df_mod3['GrLivArea'] > ar]
    try:
        fig4_tmp = alt.Chart(df_mod3).mark_point().encode(x='GrLivArea',y='SalePrice')
        fig4 = fig4_tmp + fig4_tmp.transform_regression('GrLivArea','SalePrice').mark_line()
        st.altair_chart(fig4, use_container_width=True)
    except:
        print("No linear regression found between the two.")
    ## Common Ammenities
    st.markdown("##### A pool, basement, and garage are amenities that many buyers value highly. Details about them are provided below:")
    tab1, tab2, tab3 = st.tabs(["Basement", "Pool", "Garage"])
    with tab1:
        st.markdown("##### Would you be interested in a Basement? ")
        st.image("https://wetbasements.com/wp-content/uploads/2018/02/basement.jpg.webp", width=200)
        try:
            if (df_mod3['TotalBsmtSF'].sum())!= 0:
                st.write("If this feature appeals to you, the plot below will give you an estimate of how much a property with this feature will cost, at the very least.")
                fig5 = plt.figure(figsize=(5,5))
                sns.boxplot(x='BsmtQual',y='SalePrice',data=df_mod3, order=["Ex", "Gd", "TA", "Fa", "Po", "NA"])
                st.pyplot(fig5)
            else:
                st.caption("Sorry!! No basements for properties under your selection.")
        except:
            print("No linear regression found between the two.")
    with tab2:
        st.markdown("##### How about a Pool?")
        st.image("https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/swimming-pool-designs-luxury-1651197615.png?crop=1.00xw:0.718xh;0,0.170xh&resize=980:*", width=200)
        try:
            if (df_mod3['PoolArea'].sum())!= 0:
                st.write("You may get an idea of the pricing range for a home with a pool from the plot below.")
                fig5 = plt.figure(figsize=(5,5))
                sns.boxplot(x='PoolQC',y='SalePrice',data=df_mod3, order=["Ex", "Gd", "TA", "Fa", "Po", "NA"])
                st.pyplot(fig5)
            else:
                st.caption("Sorry! No Pools in your selection")
        except:
            print("No linear regression found between the two.")
    with tab3:
        st.markdown("##### There should definetly be a garage !")
        st.image("https://images2.dwell.com/photos/6691330331594723328/6788049719024275456/original.jpg?auto=format&q=35&w=1440", width=200)
        try:
            if (df_mod3['GarageCars'].sum())!= 0:
                st.write("A garage-equipped home is a nice bonus, and this plot will show you how much you'll need to spend, at the very least, if you would like one.")
                fig5 = plt.figure(figsize=(5,5))
                sns.boxplot(x='GarageQual',y='SalePrice',data=df_mod3, order=["Ex", "Gd", "TA", "Fa", "Po", "NA"])
                st.pyplot(fig5)
            else:
                st.caption("Sorry !! No Garage in properties under your selection.")
        except:
            print("No linear regression found between the two.")
    st.write("")
    st.write("")
    st.write("")
    ## Exploration of these ammenities
    st.markdown("##### Would you like to know more about the above mentioned ammenities?")
    opt = st.radio("Which feature would like to explore more?",
        ('Basement', 'Pool', 'Garage'))
    if opt == 'Basement':
        try:
            if (df_mod3['TotalBsmtSF'].sum())!= 0:
                st.write("The distribution of the basement's total square footage according to its conditions is shown in the plot below.")
                fig6 = plt.figure(figsize=(5,5))
                df_mod4 = df_mod3.loc[df_mod3['BsmtQual'] != 'Not Applicable']
                sns.histplot(data=df_mod4, y='TotalBsmtSF',hue='BsmtQual', hue_order=["Ex", "Gd", "TA", "Fa", "Po"], multiple='stack')
                st.pyplot(fig6)
            else:
                st.caption("Sorry!! No basements for properties under your selection.")
        except:
            print("No distribution graph for basement.")
    elif opt == 'Pool':
        try:
            if (df_mod3['PoolArea'].sum())!= 0:
                st.write("The distribution of the Pool's total square footage according to its quality is shown in the plot below.")
                fig6 = plt.figure(figsize=(5,5))
                df_mod4 = df_mod3.loc[df_mod3['PoolQC'] != 'Not Applicable']
                sns.histplot(data=df_mod4, x='PoolArea',hue='PoolQC', hue_order=["Ex", "Gd", "TA", "Fa", "Po"], multiple='stack')
                st.pyplot(fig6)
            else:
                st.caption("Sorry! No Pools in your selection")
        except:
            print("No distribution graph for Pool.")
    elif opt == 'Garage':
        try:
            if (df_mod3['GarageCars'].sum())!= 0:
                st.write("The distribution of the size of the garage in car capacity stacked by their quality is shown in the plot below.")
                fig6 = plt.figure(figsize=(5,5))
                df_mod4 = df_mod3.loc[df_mod3['GarageQual'] != 'Not Applicable']
                sns.histplot(data=df_mod4, x='GarageCars',hue='GarageQual', hue_order=["Ex", "Gd", "TA", "Fa", "Po"], multiple='stack')
                st.pyplot(fig6)
            else:
                st.caption("Sorry !! No Garage in properties under your selection.")
        except: 
            print("No distribution graph for Garage.")
    st.write("")
    st.write("")
    st.write("")
    ## Payment Option
    st.write("##### The above features now give you an idea about the sale prices of various properties with varying features.")
    st.write("Understanding a property's type of sale is beneficial. To give you an indication of how well prepared you should be in terms of getting your funds ready for purchase, the plot below illustrates the count of sale type for the listed properties.")
    val = df['SaleType'].value_counts()
    try:
        fig7 = plt.figure(figsize=(5,5))
        clrs = ['grey' if (x < max(val)) else 'red' for x in val ]
        sns.countplot(data=df, x='SaleType',palette=clrs)
        st.pyplot(fig7)
    except:
        print("Error in Sale Type Graph.")
    st.write("As seen in the plot above, the Conventional Warranty Deed, abbreviated as WD, is the most well-known and preferred type of transaction.")


if page == "Prediction Model":
    st.header('Welcome to the Machine Learning part of the House Sales Project !!!')
    img4 = "https://imageio.forbes.com/specials-images/dam/imageserve/966248982/960x0.jpg?format=jpg&width=960"
    img3 = "https://data-science-blog.com/wp-content/uploads/2022/05/linear-regression-error-term.png"
    colA, colB = st.columns(2)
    colA.image(img4, use_column_width=True)
    colB.image(img3, use_column_width=True)
    data = pd.read_csv("Real_Estate_Clean.csv")
    Xn = data.drop(columns=['saleprice'])
    y = data['saleprice']
    X_train, X_test, y_train, y_test = train_test_split(Xn, y, random_state = 42, test_size=.2, train_size = .8)
    
    st.markdown('''<p style="font-family:Courier; text-align: justify; color:Blue; font-size: 20px; 
    font-weight: 650;">Please use the below filters to get the exact cost of your dream home.''',unsafe_allow_html=True)
    st.write("Note: If no selections are made, it simply takes the default values.")
    col1, col2= st.columns(2)
    age = col1.slider("Neighboorhood Quality Index", min_value=int(0), max_value=int(136))
    neigh_qual = col2.slider("Neighboorhood Quality Index", min_value=float(1.0), max_value=float(4.0))
    local_feature = st.slider("Proximity to ammenities (10 being the nearest)", min_value=float(0.0), max_value=float(4.0))
    col3,col4 = st.columns(2)
    col_Val= col3.selectbox("Will remodelled building be ok?",("Yes","No"))
    if col_Val == "Yes":
        remodel = 1
    else:
        remodel = 0
    single=0
    multi=0
    house1=0
    house2=0
    house3=0
    building = col4.selectbox("Type of Building",("Single Story","Multi Story", "Middle Unit Townhouse", "End Unit Townhouse", "Family House"))
    if building == "Single Story":
        single = 1
    elif building == "Multi Story":
        multi = 1
    elif building == "Middle Unit Townhouse":
        house1=0
    elif building == "End Unit Townhouse":
        house2=0
    elif building == "Family House":
        house3=0
    col5, col6 = st.columns(2)
    overall_qual = col5.slider("Overall Building Quality", min_value=float(1.0), max_value=float(10.0))
    outside = col6.slider("Grill Space in the Varanda", min_value=float(0.0), max_value=float(10.0))
    col7, col8 = st.columns(2)
    exterior = col7.slider("Exterior Quality Index",float(1.0), max_value=float(5.0))
    external_feature = 2.0
    roof_Val = col8.radio("Do you like a hip roof?",("Yes","No"), horizontal=True)
    if roof_Val == "Yes":
        roof=1
    else:
        roof=0
    col9,col10 = st.columns(2)
    mason = col9.slider("Masonary Veneer Area in sqft", min_value=int(0), max_value=int(1600))
    functional = 0
    lot_frontage = col10.slider("Lot Frontage", min_value=int(0), max_value=int(400))
    col11, col12 = st.columns(2)
    lot_Area = col11.slider("Total Lot Area", min_value=int(1300), max_value=int(159000))
    garage = col12.slider("Garage Space",min_value=int(0), max_value=int(4068))
    col13,col14 = st.columns(2)
    extra_car=0
    paved_drive=0
    bmt = col13.slider("Basement Quality",min_value=float(0.0), max_value=float(5.0))
    bmt_area = col14.slider("Basement Area",min_value=float(0.0), max_value=float(12948.0))
    bmt_Exposure = 1
    col15,col16 = st.columns(2)
    heater_qual = col15.slider("Heater Quality",min_value=float(0.0), max_value=float(5.0))
    kitchen_qual = col16.slider("Kitchen Quality", min_value=float(0.0), max_value=float(5.0))
    qual_room = 250
    col17,col18,col19 = st.columns(3)
    fireplace_qu = col17.slider("Fireplace Quality", min_value=float(0.0), max_value=float(5.0))
    room_num = col18.slider("Number of Rooms",min_value=int(2), max_value=int(14))
    room_size = col19.slider("Size of the master bedroom",min_value=float(100.0), max_value=float(530.0))

    data = np.array([age, neigh_qual, local_feature, remodel, overall_qual, single, multi, exterior,external_feature,
    house1, house2, house3,roof, mason,functional, lot_frontage,lot_Area, outside,
    garage, extra_car, paved_drive, bmt, bmt_area, bmt_Exposure, heater_qual, kitchen_qual,
    fireplace_qu, qual_room, room_size, room_num]).reshape(1, -1)
    sc = StandardScaler()
    Xx_test = sc.fit_transform(X_test)
    Xx_train = sc.fit_transform(X_train)
    def linear_model():
        lr = LinearRegression()
        lr.fit(Xx_train, y_train)
        cross_validation_score = round(cross_val_score(lr, Xx_train, y_train).mean(),4)
        return lr,cross_validation_score

    def lasso_linear_model(al,n):
        lasso_coef_eval = Lasso(alpha = al,  max_iter=n)
        lasso_coef_eval.fit(Xx_train, y_train)
        cross_validation_score = round(cross_val_score(lasso_coef_eval, Xx_train, y_train).mean(),4)
        return lasso_coef_eval,cross_validation_score

    def DT_regressor(depth):
        regressor = DecisionTreeRegressor(max_depth=depth,random_state=0)
        regressor.fit(Xx_train, y_train)
        cross_validation_score = round(cross_val_score(regressor, Xx_train, y_train).mean(),4)
        return regressor,cross_validation_score
    
    def SVR_model(c_val):
        svr_model = SVR(C=c_val)
        svr_model.fit(Xx_train, y_train)
        cross_validation_score = round(cross_val_score(svr_model, Xx_train, y_train).mean(),4)
        return svr_model,cross_validation_score


    st.subheader("Which algorithm will you like to use ?")
    classifier = st.selectbox("",("Linear Regression","Decision Trees","Support Vector Regression"))
    if classifier == "Linear Regression":
        st.write("A dependent variable's nature and degree of connection with a group of independent variables are assessed using linear regression which aids in the creation of models for making predictions.")
        st.write("LASSO (Least Absolute Shrinkage and Selection Operator): In order to improve the predictability and interpretability of the produced statistical model, this regression analysis technique is used. It performs both variable selection and regularization. The concept is that the data values are shrunk towards a central point, like the mean.")
        val0 = st.radio("Would you like to implement the lasso feature?",("Yes","No"))
        if val0 == "Yes":
            al = st.slider("Select Alpha Value", min_value=float(0.1), max_value=float(200))
            n = st.slider("How many itterations would you like? ",min_value=int(1000), max_value=int(8000))
            model1,score1 = lasso_linear_model(al,n)
            predicted_price = round(model1.predict(data)[0],4)
            if st.button("Click to find out the House Price in USD"):
                st.write(predicted_price)
            st.write("NOTE:  Cross Validation Score is an estimate to determine the skill of a model, that is to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.")
            if st.button("Cross Validation Score"):
                st.write(score1)

        elif val0 == "No":
            model2,score2 = linear_model()
            predicted_price = round(model2.predict(data)[0],4)
            if st.button("Click to find out the House Price in USD"):
                st.write(predicted_price)
            st.write("NOTE:  Cross Validation Score is an estimate to determine the skill of a model, that is to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.")
            if st.button("Cross Validation Score"):
                st.write(score2)

    if classifier == "Decision Trees":
        st.write("Decision Tree Classifier")
        st.write("How it works?")
        st.write("A decision tree is generated when each decision node in the tree contains a test on some input variable's value. The terminal nodes of the tree contain the predicted output variable values.")
        img5 = "https://www.saedsayad.com/images/Decision_tree_r1.png"
        st.image(img5, use_column_width=True)
        depth = st.slider("Please Input the maximum depth of the tree you would like to apply on the algorithm : ",min_value=int(1),max_value=int(40))
        model3,score3 = DT_regressor(depth)
        predicted_price = round(model3.predict(data)[0],4)
        if st.button("Click to find out the House Price in USD"):
            st.write(predicted_price)
        st.write("NOTE:  Cross Validation Score is an estimate to determine the skill of a model, that is to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.")
        if st.button("Cross Validation Score"):
            st.write(score3)

    if classifier == "Support Vector Regression":
        img6 = "https://cdn.educba.com/academy/wp-content/uploads/2020/01/Support-Vector-Regression.jpg"
        st.image(img6,use_column_width=True)
        st.write("Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line. In SVR, the best fit line is the hyperplane that has the maximum number of points.")
        st.write("C parameter is the penalty parameter of the error term. For each incorrectly calculated data point, it essentially imposes a penalty. When c is small, the penalty is low, leading to the selection of a decision boundary with a high margin at the expense of more incorrect predictions. SVM attempts to reduce the number of incorrect predictions caused by high penalty when c is big, which results in a decision boundary with a narrower margin.")
        c_val = st.slider("Please input the C Value",min_value=float(0.1),max_value=float(40.0))
        model4,score4 = SVR_model(c_val)
        predicted_price = round(model4.predict(data)[0],4)
        if st.button("Click to find out the House Price in USD"):
            st.write(predicted_price)
        st.write("NOTE:  Cross Validation Score is an estimate to determine the skill of a model, that is to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.")
        if st.button("Cross Validation Score"):
            st.write(score4)
    







    