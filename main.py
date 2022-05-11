# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn import datasets
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, plot_confusion_matrix
#import xgboost
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
import lime
import lime.lime_tabular
import shap
import altair as alt
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.plotting import plot_confusion_matrix as plot_cm

#Function to display sidebar
def display_sidebar(dataFeatures, datasetType):
    featureList = {}

    if datasetType == 'cluster':
        for i, name in enumerate(dataFeatures):

            if name == 'n_samples':
                value = st.sidebar.slider(name, 100, 10000, 500)
            elif name == 'n_features':
                value = st.sidebar.slider(name, 2, 10, 2)
            elif name == 'num_centers':
                value = st.sidebar.slider(name, 2, 50, 3)
            elif name == 'random_state':
                value = st.sidebar.slider(name, 2, 100, 42)
            elif name == 'return_centers':
                value = st.sidebar.checkbox(name, 'True', on_change = 'False')

            featureList[name] = value

    if datasetType == 'make_class':
        for i, name in enumerate(dataFeatures):

            if name == 'n_samples':
                value = st.sidebar.slider(name, 100, 10000, 500)
            elif name == 'n_features':
                value = st.sidebar.slider(name, 2, 20, 2)
            elif name == 'n_informative':
                value = st.sidebar.slider(name, 0, 20, 2)
            elif name == 'n_redundant':
                value = st.sidebar.slider(name, 0, 20, 0)
            elif name == 'n_repeated':
                value = st.sidebar.slider(name, 0, 20, 0)
            elif name == 'flip_y':
                value = st.sidebar.slider('Choose proportion of Noise [0-1]: ', 0.0, 1.0, 0.0)
            elif name == 'weight':
                value = st.sidebar.selectbox('Class Imbalance', ['Balanced', 'Imbalanced'])

            featureList[name] = value

    if datasetType == 'gaussian':
        for i, name in enumerate(dataFeatures):

            if name == 'n_samples':
                value = st.sidebar.slider(name, 100, 10000, 500)
            #elif name == 'n_features':
            #    value = st.sidebar.slider(name, 2, 10, 2)
            elif name == 'n_classes':
                value = st.sidebar.slider(name, 2, 50, 3)
            elif name == 'cov':
                value = st.sidebar.slider(name, 0.000, 50.0, 3.0)

            featureList[name] = value

    return featureList

#function to create synthetic data from a list of parameters
def getSyntheticClusterDataset():

    st.markdown('## Create Synthetic Data')

    # Display the features in a sidebar
    st.sidebar.markdown("## Create Data using the below features")

    #initialize feature names
    dataFeatures = ['n_samples',  # =500
                    'n_features',  # =2
                    'num_centers',  # =3
                    'random_state',  # =1
                    'return_centers']  # =True]

    # get chosen features
    parameters = display_sidebar(dataFeatures, 'cluster')


    if parameters['return_centers']:
        x, y, centers = make_blobs(n_samples=parameters['n_samples'],
                                   n_features=parameters['n_features'],
                                   centers=parameters['num_centers'],
                                   random_state=parameters['random_state'],
                                   return_centers=parameters['return_centers'])
        #Construct Dataframe
        df = pd.DataFrame(x)
        df['label'] = y

        #print(list(df.columns[-1:]))

        #Change Column Names
        dataColumns = list(df.columns[:-1])
        for i, col in enumerate(dataColumns):
            dataColumns[i] = 'X' + str(col)

        df.columns = dataColumns + ['label']

        return df, centers
    else:
        x, y = make_blobs(n_samples=parameters['n_samples'],
                                   n_features=parameters['n_features'],
                                   centers=parameters['num_centers'],
                                   random_state=parameters['random_state'],
                                   return_centers=parameters['return_centers'])
        # Construct Dataframe
        df = pd.DataFrame(x)
        df['label'] = y

        # Change Column Names
        dataColumns = list(df.columns[:-1])
        for i, col in enumerate(dataColumns):
            dataColumns[i] = 'X' + str(col)

        df.columns = dataColumns + ['label']

        return df

#function to load Census Data
def getCensusData():
    df = pd.read_csv("census-income.data", na_values=' ?', header=None)
    # mapping data to column names
    df.columns = [
        "age",
        "class_of_worker",
        "detailed_industry_recode",
        "detailed_occupation_recode",
        "education",
        "wage_per_hour",
        "enroll_in_edu_inst_last_wk",
        "marital_stat",
        "major_industry_code",
        "major_occupation_code",
        "race",
        "hispanic_origin",
        "sex",
        "member_of_a_labor_union",
        "reason_for_unemployment",
        "full_or_part time employment stat",
        "capital_gains",
        "capital_losses",
        "dividends_from_stocks",
        "tax_filer_stat",
        "region_of_previous_residence",
        "state_of_previous_residence",
        "detailed_household_and_family_stat",
        "detailed_household_summary_in_household",
        "instance_weight",
        "migration_code_change_in_msa",
        "migration_code_change in reg",
        "migration_code_move_within_reg",
        "live_in_this_house_1_year_ago",
        "migration_prev_res_in_sunbelt",
        "num_persons_worked_for_employer",
        "family_members_under_18",
        "country_of_birth_father",
        "country_of_birth_mother",
        "country_of birth_self",
        "citizenship",
        "own_business_or_self_employed",
        "fill_inc_questionnaire_for_veteran's_admin",
        "veterans_benefits",
        "weeks_worked_in_year",
        "year",
        "under_50k_over_50k"
    ]
    # drop columns having missing values
    df = df.dropna(axis=1)
    df = df.reset_index(drop=True)
    # getting a list of categorical columns
    objList = df.select_dtypes(include="object").columns
    # Label Encoding for object to numeric conversion
    le = LabelEncoder()

    for feat in objList:
        df[feat] = le.fit_transform(df[feat].astype(str))

    print(df)

    return df, le, objList

def getMakeClassificationData():
    # Display the features in a sidebar
    st.sidebar.markdown("## Create Data using the below features")

    # initialize feature names
    dataFeatures = ['n_samples',
                    'flip_y',
                    'n_features',
                    'n_informative',
                    'n_redundant',
                    'n_repeated',
                    'weight']

    # get chosen features
    parameters = display_sidebar(dataFeatures, 'make_class')

    # get class imbalance choice
    if parameters['weight'] == 'Balanced':
        weights = [0.5, 0.5]
    else:
        weights = [0.8, 0.2]

    # print(data)
    x, y = make_classification(n_samples=parameters['n_samples'],
                               n_features=parameters['n_features'],
                               n_informative=parameters['n_informative'],
                               n_redundant=parameters['n_redundant'],
                               n_repeated=parameters['n_repeated'],
                               n_classes=2,
                               n_clusters_per_class=1,
                               class_sep=2,
                               flip_y=parameters['flip_y'],
                               weights=weights,
                               random_state=42)

    # Construct Dataframe
    df = pd.DataFrame(x)
    df['label'] = y

    # Change Column Names
    dataColumns = list(df.columns[:-1])
    for i, col in enumerate(dataColumns):
        dataColumns[i] = 'X' + str(col)

    df.columns = dataColumns + ['label']

    return df

def getGuassianData():
    # Display the features in a sidebar
    st.sidebar.markdown("## Create Gaussian data using the below features")

    # initialize feature names
    dataFeatures = ['cov',
                    'n_samples',
                    'n_classes',
                    'n_features']

    # get chosen features
    parameters = display_sidebar(dataFeatures, 'gaussian')

    # print(data)
    x, y = make_gaussian_quantiles(cov= parameters['cov'],#3.,
                                   n_samples = parameters['n_samples'],
                                   n_features = 2,#parameters['n_features'],
                                   n_classes = parameters['n_classes'],
                                   random_state=42)

    # Construct Dataframe
    df = pd.DataFrame(x)
    df['label'] = y

    # Change Column Names
    dataColumns = list(df.columns[:-1])
    for i, col in enumerate(dataColumns):
        dataColumns[i] = 'X' + str(col)

    df.columns = dataColumns + ['label']

    return df


#get selected columns
def get_plotting_data(df, selected_tickers):
    #data = df[selected_tickers+['label']]#.query("ticker in {}".format(selected_tickers))
    data = df[selected_tickers + list(df.columns[-1:])]
    #data = data['close']
    return data


def get_plot_figure(data):
    #fig = plt.Figure()

    x = data[data.columns[:-1]]
    #y = data['label']
    y = data[data.columns[-1:]]

    fig, ax = plt.subplots()

    #1-D Plot
    if x.shape[1] == 1:
        chart = (
            alt.Chart(data=data).mark_bar().encode(
                x = alt.X(x.columns[0], bin=True, axis=alt.Axis(title=x.columns[0])),
                y = alt.X('count()'),
                #y=alt.X(x.columns[1], axis=alt.Axis(title=x.columns[1])),
                color= y.columns[0],# 'label',
                tooltip=[x.columns[0]]
            ).interactive()
        )


    #2-D Plot
    elif x.shape[1] == 2:
        chart = (
            alt.Chart(data=data).mark_circle().encode(
                x=alt.X(x.columns[0], axis=alt.Axis(title=x.columns[0])),
                y=alt.X(x.columns[1], axis=alt.Axis(title=x.columns[1])),
                color=y.columns[0], # 'label'
                tooltip=[x.columns[0], x.columns[1]]
            ).interactive()
        )

    #3-D Plot
    else:
        chart = plt.figure(figsize=(12, 9))
        ax = Axes3D(chart)

        for g in np.unique(y):
            i = np.where(y == g)

            ax.scatter(*x.iloc[i[0]].T.values, label=g)

        ax.set_xlabel(x.columns[0])
        ax.set_ylabel(x.columns[1])
        ax.set_zlabel(x.columns[2])

    return chart


def getModelName():
    option = st.selectbox(
        'Which model would you like to train?',
        ('Logistic Regression', 'SVM Classifier', 'MLP Classifier', 'Decision Tree Classifier', 'Random Forest Classifier'))

    return option

def init_model(modelName):

    if modelName == 'SVM Classifier':
        hyperparameters = ['kernel', 'C', 'gamma', 'degree']
        hyperOptions = {}
        for name in hyperparameters:
            if name == 'kernel':
                value = st.selectbox(
                    name,
                    ('linear', 'rbf', 'sigmoid', 'poly'))
            elif name == 'C':
                value = st.slider(name, 0, 10, 1)
            elif name == 'gamma':
                value = st.selectbox(
                    name,
                    ('scale', 'auto', 0.0001, 0.001, 0.01, 1, 10, 100, 1000))
            elif name == 'degree':
                value = st.slider(name, 2, 5, 3)

            hyperOptions[name] = value

        model = SVC(**hyperOptions, random_state=42, probability=True)

        return model

    elif modelName == 'MLP Classifier':
        hyperparameters = ['activation', 'solver', 'learning_rate', 'max_iter']
        hyperOptions = {}
        for name in hyperparameters:
            if name == 'activation':
                value = st.selectbox(
                    name,
                    ('relu', 'identity', 'logistic', 'tanh'))
            elif name == 'solver':
                value = st.selectbox(
                    name,
                    ('adam', 'sgd', 'lbfgs'))
            elif name == 'learning_rate':
                value = st.selectbox(
                    name,
                    ('constant', 'invscaling', 'adaptive'))
            elif name == 'max_iter':
                value = st.slider(name, 10, 500, 100)

            hyperOptions[name] = value

        model = MLPClassifier(**hyperOptions, random_state=42, learning_rate_init=0.001)

        return model

    elif modelName == 'Logistic Regression':
        hyperparameters = ['penalty', 'solver', 'C', 'max_iter', 'fit_intercept', 'multi_class']
        hyperOptions = {}
        for name in hyperparameters:
            if name == 'penalty':
                value = st.selectbox(
                    name,
                    ('l2', 'l1', 'none', 'elasticnet'))
            elif name == 'solver':
                value = st.selectbox(
                    name,
                    ('lbfgs', 'newton-cg', 'sag', 'saga'))
            elif name == 'C':
                value = st.slider(name, 0.01, 10.0, 1.0)
            elif name == 'max_iter':
                value = st.slider(name, 10, 500, 100)
            elif name == 'fit_intercept':
                value = st.selectbox(
                    name,
                    (True, False))
            elif name == 'multi_class':
                value = st.selectbox(
                    name,
                    ('auto', 'ovr', 'multinomial'))

            hyperOptions[name] = value

        model = LogisticRegression(**hyperOptions, random_state=42)

        return model

    elif modelName == 'Decision Tree Classifier':
        hyperparameters = ['criterion', 'max_depth', 'min_samples_leaf', 'max_features']
        hyperOptions = {}
        for name in hyperparameters:
            if name == 'criterion':
                value = st.selectbox(
                    name,
                    ('gini', 'entropy'))
            elif name == 'max_depth':
                value = st.slider(name, 1, 10, 1)
            elif name == 'min_samples_leaf':
                value = st.slider(name, 1, 10, 1)
            elif name == 'max_features':
                value = st.slider(name, 1, 100, 1)

            hyperOptions[name] = value

        model = DecisionTreeClassifier(**hyperOptions, random_state=42)
        return model

    elif modelName == 'Random Forest Classifier':
        hyperparameters = ['n_estimators', 'criterion', 'max_depth', 'min_samples_leaf', 'max_features']
        hyperOptions = {}
        for name in hyperparameters:
            if name == 'n_estimators':
                value = st.slider(name, 1, 200, 10)
            elif name == 'criterion':
                value = st.selectbox(
                    name,
                    ('gini', 'entropy'))
            elif name == 'max_depth':
                value = st.slider(name, 1, 100, 10)
            elif name == 'min_samples_leaf':
                value = st.slider(name, 10, 500, 10)
            elif name == 'max_features':
                value = st.slider(name, 1, 100, 2)

            hyperOptions[name] = value

        model = RandomForestClassifier(**hyperOptions, random_state=42)

        return model

def printClassificationReport(model, X_test, y_test):
    #get predicted labels
    y_preds_label = model.predict(X_test)

    #get class report
    report = classification_report(y_test, y_preds_label)  # , target_names=np.unique(df['label']))

    st.write(report)

    #calculate confusion matrix
    conf = confusion_matrix(y_preds_label, y_test)

    figure, ax = plot_cm(conf_mat=conf,
                         # class_names=classes,
                         show_absolute=True,
                         show_normed=False  # ,
                         # colorbar=True
                         )

    st.pyplot(figure)

def getLimeExplanations(model, X_train, X_test, y_train, y_test, dfOld, datasetName):
    hyperparameters = ['num_samples', 'num_features', 'discretize_continuous', 'discretizer', 'verbose']
    hyperOptions = {}
    st.markdown('### Choose Lime Hyperparameters')

    if datasetName == 'Census (Case Study)':
        defNumSamples = 500
        defNumFeatures = 15
    else:
        defNumSamples = 2
        defNumFeatures = 3

    for name in hyperparameters:
        if name == 'num_samples':
            value = st.slider(name, 1, X_train.shape[0], defNumSamples)
        elif name == 'num_features':
            value = st.slider(name, 1, X_train.shape[1], defNumFeatures)
        elif name == 'discretize_continuous':
            value = st.selectbox(
                    name,
                    (True, False))
        elif name == 'discretizer':
            value = st.selectbox(
                    'decretizer (only if descretize_continuous -> True)',
                    ('quartile', 'decile', 'entropy'))
        elif name == 'verbose':
            value = st.selectbox(
                name,
                (True, False))

        hyperOptions[name] = value

    #get the instance to explain
    instance = st.slider('Choose the instance to explain:', 0, X_test.shape[0], 0)

    # print(np.unique(y_test))

    # Train an LIME explainer
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                                                            feature_names=X_test.columns,
                                                            class_names=np.unique(y_test),#['0', '1', '2'],
                                                            discretize_continuous = hyperOptions['discretize_continuous'],
                                                            discretizer=hyperOptions['discretizer'],
                                                            # Non categorical features will be put into buckets (discretized)
                                                            verbose=hyperOptions['verbose'],
                                                            mode='classification',
                                                            random_state=42
                                                            )

    # Display distribution of labels

    # testIndices = np.array(range(y_test.shape[0]))
    #
    # fig = plt.figure(figsize = (20, 10))
    # fig.bar(testIndices, np.array(y_test[y_test.columns[0]]))
    # st.pyplot()

    #Output the Selected Instance
    st.write('Instance: ', dfOld.iloc[X_test.iloc[instance:instance+1].index])

    exp = explainer_lime.explain_instance(X_test.iloc[instance],
                                          model.predict_proba,
                                          num_features=hyperOptions['num_features'],
                                          num_samples=hyperOptions['num_samples'],
                                          labels = np.unique(y_test),
                                          top_labels = len(np.unique(y_test))
                                          )

    #get class probabilities
    predictProbs = model.predict_proba(np.array(X_test.iloc[instance,:]).reshape(1, -1))
    #get label with highest probability
    label = np.argmax(predictProbs)

    # plot lime explaination for particular class
    # for i in range(len(np.unique(y_test))):
    classLabel = st.selectbox('Select Class: ', np.unique(y_test), int(label))
    st.write('Class ' + str(classLabel) + ' Probability: ', predictProbs[0, classLabel])
    st.write('Class ' + str(classLabel) + ' Lime Explanations: ', exp.as_list(label = classLabel))
    exp.as_pyplot_figure(label = classLabel)
    st.pyplot()

    #print(hyperOptions)

    #print(np.unique(y_test))

    #print(np.array(X_test.iloc[0]))

    # print(exp.as_list(label=0))
    # print(exp.as_list(label=1))
    # print(exp.as_list(label=2))
    # print(model.predict_proba(np.array(X_test.iloc[instance,:]).reshape(1, -1)))

#helper function to plot js plots in streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

#Function for plotting Shap Summary
def plotShapSummary(shap_values, X_test):
    # Select number of samples to show Summary of
    summarySlice = st.slider('Select number of Samples to Create Shap Explanations with', 1, X_test.shape[0], 100)

    st.markdown('### Shap Summary')
    shap.summary_plot(shap_values, X_test[:summarySlice], plot_type='bar')
    st.pyplot()

#Function for Shap Dependancy Plot
def plotShapDependency(shap_values, X_test, y_test):
    # Select upto 2 features for dependence plot
    dependanceFeatures = st.multiselect(
        "Choose upto 2 features that you would like to plot:",
        X_test.columns,
        [X_test.columns[0]]
    )

    # Render dependency plot with option to select class
    if len(dependanceFeatures) > 2:
        st.error("Please select up to 2 features only!")
    elif len(dependanceFeatures) == 0:
        st.error("Please select at least one Column to plot!")
    elif len(dependanceFeatures) == 1:
        # for i in range(len(np.unique(y_test))):
        classLabel = st.selectbox('Select the Class:', np.unique(y_test))
        st.write('Shap Dependancy Plot for feature ' + dependanceFeatures[0] + ', Class: ' + str(classLabel) + '')
        shap.dependence_plot(dependanceFeatures[0], shap_values[classLabel], X_test, interaction_index=None)
        st.pyplot()
    elif len(dependanceFeatures) == 2:
        # for i in range(len(np.unique(y_test))):
        classLabel = st.selectbox('Select the Class:', np.unique(y_test))
        st.write('Shap Dependancy Plot for features ' + dependanceFeatures[0] + ' vs ' + dependanceFeatures[
            1] + ', Class: ' + str(classLabel))
        shap.dependence_plot(dependanceFeatures[0], shap_values[classLabel], X_test,
                             interaction_index=dependanceFeatures[1])
        st.pyplot()

#Function for Shap Waterfall plot
def plotShapWaterfall(shap_values, explainer, X_test, y_test, model):
    # Select Instance and Class for waterfall plot
    waterfallInstance = st.slider('Select Instance for Waterfall Plot:', 0, X_test.shape[0], 0)

    # get class probabilities
    predictProbs = model.predict_proba(np.array(X_test.iloc[waterfallInstance, :]).reshape(1, -1))
    # get label with highest probability
    label = np.argmax(predictProbs)

    waterfallInstanceClass = st.selectbox('Select Class for Waterfall Plot:', np.unique(y_test), int(label))

    # Define New Shap Class to get data in Waterfall format
    class shapNew():
        def __init__(self, a, b, c, d):
            self.values = a
            self.base_values = b
            self.data = c
            self.feature_names = d

    # Initialize Class object with selected instance and class
    shap_values_new = shapNew(shap_values[waterfallInstanceClass][waterfallInstance],
                              explainer.expected_value[waterfallInstanceClass],
                              np.array(X_test.iloc[waterfallInstance]),
                              X_test.columns)

    # render Waterfall plot
    shap.plots.waterfall(shap_values_new)
    #fig.title('Waterfall Plot for Instance ', str(waterfallInstance), ' Class ', str(waterfallInstanceClass))
    st.pyplot()

#Function for Shap Force Plot
def plotForce(shap_values, explainer, X_test, y_test, model, single = True):

    if single:
        # Select Instance and Class
        forceInstance = st.slider('Select Instance for Force Plot:', 0, X_test.shape[0], 0)

        # get class probabilities
        predictProbs = model.predict_proba(np.array(X_test.iloc[forceInstance, :]).reshape(1, -1))
        # get label with highest probability
        label = np.argmax(predictProbs)

        forceClass = st.selectbox('Select Class for Force Plot:', np.unique(y_test), int(label))

        # matplotlib plot
        shap.force_plot(explainer.expected_value[forceClass], shap_values[forceClass][forceInstance], matplotlib=True,
                        figsize=(20, 7))
        st.pyplot()
        # Javascript Plot
        # st_shap(shap.force_plot(explainer.expected_value[forceClass], shap_values[forceClass][forceInstance]), 400)

    else:
        # Select Class
        forceClassAllTest = st.selectbox('Select Class for Complete Force Plot:', np.unique(y_test))
        # matplotlib plot
        # shap.force_plot(explainer.expected_value[forceClass], shap_values[forceClass], X_test, matplotlib=True,
        #                 figsize=(20, 7))
        # st.pyplot()
        # Javascript Plot
        st_shap(shap.force_plot(explainer.expected_value[forceClassAllTest],
                                shap_values[forceClassAllTest], X_test),
                400)
@st.cache()
def returnShapExpVal(model, X_train, X_test, hyperOptions):
    explainerShap = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, hyperOptions['ShapSamples']))
    shap_values = explainerShap.shap_values(X_test, n_samples=hyperOptions['n_samples'])
    return shap_values, explainerShap

def getShapExplanations(model, X_train, X_test, y_train, y_test, datasetName):
    hyperparameters = ['ShapSamples', 'n_samples', 'discretize_continuous']
    hyperOptions = {}
    st.markdown('### Choose Shap Hyperparameters')

    if datasetName == 'Census (Case Study)':
        defShapSamples = 100
        defNSamples = 15
    else:
        defShapSamples = 10
        defNSamples = 10

    for name in hyperparameters:
        if name == 'ShapSamples':
            value = st.slider('Select number of Samples to Create Shap Explanations with', 10, X_train.shape[0], defShapSamples)
        elif name == 'n_samples':
            value = st.slider(name, 2, X_test.shape[0], defNSamples)
        # elif name == 'discretize_continuous':
        #     value = st.selectbox(
        #         name,
        #         (True, False))

        hyperOptions[name] = value

    #Initialize Explainer and Calculate Shap Values
    # explainerShap = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, hyperOptions['ShapSamples']))
    # shap_values = explainerShap.shap_values(X_test, n_samples=hyperOptions['n_samples'])
    shap_values, explainerShap = returnShapExpVal(model, X_train, X_test, hyperOptions)


    #Plot Summary
    st.markdown('### Shap Summary')
    plotShapSummary(shap_values, X_test)

    # Shap Dependency Plot
    st.markdown('### Shap Dependency Plot')
    plotShapDependency(shap_values, X_test, y_test)

    # Plot waterfall
    st.markdown('### Waterfall Plot (Single Instance)')
    plotShapWaterfall(shap_values, explainerShap, X_test, y_test, model)

    # Force Plot (single instance)
    st.markdown('### Force Plot (Single Instance)')
    plotForce(shap_values, explainerShap, X_test, y_test, model, single = True)

    #Force Plot for all Test Predictions
    st.markdown('### Force Plot (all Test)')
    plotForce(shap_values, explainerShap, X_test, y_test, model, single = False)

def getScatter(df, dataColumns):
    st.markdown('### Scatter Plot')

    # Select Columns to plot
    selected_tickers = st.multiselect(
        "Choose the 2-3 that you would like to plot",
        dataColumns,
        [df.columns[0], df.columns[1]]
    )

    if not selected_tickers:
        st.error("Please select at least one Column to plot")
    elif len(selected_tickers) > 3:
        st.error("Please select only up to 3 Columns")
    else:
        fig = get_plot_figure(get_plotting_data(df, selected_tickers))
        if len(selected_tickers) <= 2:
            st.altair_chart(fig)
        else:
            st.pyplot(fig)

def getCorrelationMatrix(df, dataColumns, datasetName):

    st.markdown('## Correlation Matrix: ')

    if datasetName == 'Census (Case Study)':
        #select features if you want
        featuresCorr = st.multiselect('Select Correlation Features: ', df.columns[:-1],
                                      ['age', 'sex', 'education', 'capital_gains', 'dividends_from_stocks',
                                       'weeks_worked_in_year', 'capital_losses'])
    else:
        # select features if you want
        featuresCorr = st.multiselect('Select Correlation Features: ', df.columns[:-1], None)

    if len(featuresCorr)>0:
        df = df[featuresCorr + list(df.columns[-1:])]

    #print(df.head())

    corrMatrix = df.corr()
    print(corrMatrix)
    # sn.set(rc={'figure.figsize': (20, 10)})
    sn.heatmap(corrMatrix, annot=True)
    #corrMatrix.style.background_gradient(cmap='coolwarm')
    st.pyplot()

def filterData(df, datasetName, objList = None):

    st.markdown('### Filter Data (if required)')

    if datasetName == 'Census (Case Study)':
        #select features if you want
        featuresFilter = st.multiselect('Select Features to filter by: ', df.columns[:-1],
                                        ['age', 'sex', 'education', 'capital_gains', 'dividends_from_stocks',
                                         'weeks_worked_in_year', 'capital_losses'])
        objList = [col for col in objList if col in featuresFilter]
    else:
        # select features if you want
        featuresFilter = st.multiselect('Select Features to filter by: ', df.columns[:-1], None)

    if len(featuresFilter)>0:
        df = df[featuresFilter + list(df.columns[-1:])]

    # display if checkbox is checked
    checkbox = st.checkbox("Show Filtered Data")
    if checkbox:
        st.write(df.head(100))

    return df, objList



def exploreAndCompare(df, datasetName, objList = None, le = None):
    # get column names
    dataColumns = list(df.columns[:-1])
    # for i, col in enumerate(dataColumns):
    #     dataColumns[i] = 'X'+str(col)
    #
    # df.columns = dataColumns + ['label']

    # display if checkbox is checked
    checkbox = st.checkbox("Show the Data")
    if checkbox:
        st.write(df.head(100))

    # Plot Class Distribution
    #df.plot.pie(y=df.columns[-1:][0], figsize=(5, 5))
    #st.pyplot()
    counts = pd.to_numeric(df[df.columns[-1]].value_counts())
    st.write('Class Counts: ', counts/float(df.shape[0]))

    # Plot Scatter Plot
    getScatter(df, dataColumns)

    #get correlation matrix
    getCorrelationMatrix(df, dataColumns, datasetName)

    #Filter data if required
    df, objList = filterData(df, datasetName, objList)

    ## Model
    st.markdown('## Train Model')

    # get model name
    modelName = getModelName()

    # initialize model
    model = init_model(modelName)

    # scale and split data
    dfOld = df.copy()
    s = MinMaxScaler()
    df[df.columns[:-1]] = s.fit_transform(df[df.columns[:-1]])

    #change split %by dataset name
    if datasetName == 'Census (Case Study)':
        testPerc = 0.0025
    else:
        testPerc = 0.10

    X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:-1]],
                                                        df[df.columns[-1:]],
                                                        test_size=testPerc,
                                                        random_state=42)

    # train model
    model.fit(X_train, y_train)

    # print test classification report
    printClassificationReport(model, X_test, y_test)

    #Reverse Label Encoding
    #if le:
    #    for col in objList:
    #        X_train[col] = le.inverse_transform(X_train[col])
    #        X_test[col] = le.inverse_transform(X_test[col])

    # Get Lime Explanations
    st.markdown('## Lime Explanations')
    getLimeExplanations(model, X_train, X_test, y_train, y_test, dfOld, datasetName)

    # Get Shap Explanations
    st.markdown('## Shap Explanations')
    getShapExplanations(model, X_train, X_test, y_train, y_test, datasetName)

def main():

    st.title('Comparison of Lime vs Shap')
    st.markdown('### Sourabh Kumar Bhattacharjee - skb5275')
    st.markdown('### Tanmaya Sangwan - ts4370')
    #Select type of data
    dataSelection = st.selectbox('Choose the type of Data you wish to explore: ', ['Synthetic - Cluster Data',
                                                                                   'Synthetic - Class Imbalance/Noise',
                                                                                   'Synthetic - Gaussian',
                                                                                   'Census (Case Study)'])

    if dataSelection == 'Synthetic - Cluster Data':
        data = getSyntheticClusterDataset()
        #get dataframe
        df = data[0]
        #Run remaining code to explore data, train model and get explanations
        exploreAndCompare(df, dataSelection)
    elif dataSelection == 'Census (Case Study)':
        data, le, objList = getCensusData()
        #get dataframe
        df = data
        # Run remaining code to explore data, train model and get explanations
        exploreAndCompare(df, dataSelection, objList, le)
    elif dataSelection == 'Synthetic - Class Imbalance/Noise':
        data = getMakeClassificationData()
        # get dataframe
        df = data
        # Run remaining code to explore data, train model and get explanations
        exploreAndCompare(df, dataSelection)
    elif dataSelection == 'Synthetic - Gaussian':
        data = getGuassianData()
        # get dataframe
        df = data
        # Run remaining code to explore data, train model and get explanations
        exploreAndCompare(df, dataSelection)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
