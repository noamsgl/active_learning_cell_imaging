import pandas as pd
from sklearn import preprocessing, svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


def plot_pca(data_df, target):
    """plot data in 2d PCA, color by target"""
    # clear figure
    plt.clf()
    # plot
    g = sns.scatterplot(data=data_df, x='pca-2d-one', y='pca-2d-two', hue=target, palette='crest')
    g.set(xlabel="PC1", ylabel="PC2")
    plt.savefig(f"results/{target}/pca.pdf", bbox_inches="tight")


def plot_cm(X, y):
    """plot confusion matrix"""
    plt.clf()
    clf = svm.SVC()
    # train & test classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # plot confusion matrix
    plt.title("confusion matrix")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.savefig(f"results/{target}/cm.pdf", bbox_inches="tight")


def plot_decision_boundary(X, y):
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    plt.clf()
    clf = svm.SVC(random_state=0, probability=True)
    clf.fit(X, y)
    ax = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, 
            le.classes_, 
            framealpha=0.3, scatterpoints=1)
    
    plt.title("decision boundary")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(f"results/{target}/db.pdf", bbox_inches="tight")


def baseline_plots(target):
    # load data
    data_df = pd.read_csv("BCCD_Dataset/Zaretski_Image_All.csv")

    # keep only relevant rows
    if target == 'cell_type':
        classes = ['RBC', 'WBC']
        data_df = data_df.loc[data_df['cell_type'].isin(classes)]

    elif target == 'wbc_category':
        classes = ['NEUTROPHIL', 'EOSINOPHIL', 'MONOCYTE', 'LYMPHOCYTE']
        data_df = data_df.loc[data_df['wbc_category'].isin(classes)]
        
    else:
        raise ValueError(f"{target=} is not supported")

    # set features
    features_list = ['AreaOccupied_AreaOccupied_IdentifyPrimaryObjects', 'AreaOccupied_Perimeter_IdentifyPrimaryObjects', 'AreaOccupied_TotalArea_IdentifyPrimaryObjects', 'Count_IdentifyPrimaryObjects', 'ExecutionTime_03NamesAndTypes', 'ExecutionTime_06ColorToGray', 'ExecutionTime_07IdentifyPrimaryObjects', 'ExecutionTime_08MeasureGranularity', 'ExecutionTime_09MeasureImageAreaOccupied', 'ExecutionTime_10MeasureImageIntensity', 'ExecutionTime_13MeasureImageSkeleton', 'ExecutionTime_14MeasureObjectIntensity', 'ExecutionTime_15MeasureObjectIntensityDistribution', 'ExecutionTime_16MeasureObjectNeighbors', 'ExecutionTime_17MeasureObjectOverlap', 'ExecutionTime_18MeasureObjectSizeShape', 'ExecutionTime_20MeasureTexture', 'Granularity_10_OrigGray', 'Granularity_11_OrigGray', 'Granularity_12_OrigGray', 'Granularity_13_OrigGray', 'Granularity_14_OrigGray', 'Granularity_15_OrigGray', 'Granularity_16_OrigGray', 'Granularity_1_OrigGray', 'Granularity_2_OrigGray', 'Granularity_3_OrigGray', 'Granularity_4_OrigGray', 'Granularity_5_OrigGray', 'Granularity_6_OrigGray', 'Granularity_7_OrigGray', 'Granularity_8_OrigGray', 'Granularity_9_OrigGray', 'Height_Blood_Cells', 'Intensity_LowerQuartileIntensity_OrigGray', 'Intensity_MADIntensity_OrigGray', 'Intensity_MaxIntensity_OrigGray', 'Intensity_MeanIntensity_OrigGray', 'Intensity_MedianIntensity_OrigGray', 'Intensity_MinIntensity_OrigGray', 'Intensity_PercentMaximal_OrigGray', 'Intensity_StdIntensity_OrigGray', 'Intensity_TotalArea_OrigGray', 'Intensity_TotalIntensity_OrigGray', 'Intensity_UpperQuartileIntensity_OrigGray', 'Skeleton_Branches_OrigGray', 'Texture_AngularSecondMoment_OrigGray_3_00_256', 'Texture_AngularSecondMoment_OrigGray_3_01_256', 'Texture_AngularSecondMoment_OrigGray_3_02_256', 'Texture_AngularSecondMoment_OrigGray_3_03_256', 'Texture_Contrast_OrigGray_3_00_256', 'Texture_Contrast_OrigGray_3_01_256', 'Texture_Contrast_OrigGray_3_02_256', 'Texture_Contrast_OrigGray_3_03_256', 'Texture_Correlation_OrigGray_3_00_256', 'Texture_Correlation_OrigGray_3_01_256', 'Texture_Correlation_OrigGray_3_02_256', 'Texture_Correlation_OrigGray_3_03_256', 'Texture_DifferenceEntropy_OrigGray_3_00_256', 'Texture_DifferenceEntropy_OrigGray_3_01_256', 'Texture_DifferenceEntropy_OrigGray_3_02_256', 'Texture_DifferenceEntropy_OrigGray_3_03_256', 'Texture_DifferenceVariance_OrigGray_3_00_256', 'Texture_DifferenceVariance_OrigGray_3_01_256', 'Texture_DifferenceVariance_OrigGray_3_02_256', 'Texture_DifferenceVariance_OrigGray_3_03_256', 'Texture_Entropy_OrigGray_3_00_256', 'Texture_Entropy_OrigGray_3_01_256', 'Texture_Entropy_OrigGray_3_02_256', 'Texture_Entropy_OrigGray_3_03_256', 'Texture_InfoMeas1_OrigGray_3_00_256', 'Texture_InfoMeas1_OrigGray_3_01_256', 'Texture_InfoMeas1_OrigGray_3_02_256', 'Texture_InfoMeas1_OrigGray_3_03_256', 'Texture_InfoMeas2_OrigGray_3_00_256', 'Texture_InfoMeas2_OrigGray_3_01_256', 'Texture_InfoMeas2_OrigGray_3_02_256', 'Texture_InfoMeas2_OrigGray_3_03_256', 'Texture_InverseDifferenceMoment_OrigGray_3_00_256', 'Texture_InverseDifferenceMoment_OrigGray_3_01_256', 'Texture_InverseDifferenceMoment_OrigGray_3_02_256', 'Texture_InverseDifferenceMoment_OrigGray_3_03_256', 'Texture_SumAverage_OrigGray_3_00_256', 'Texture_SumAverage_OrigGray_3_01_256', 'Texture_SumAverage_OrigGray_3_02_256', 'Texture_SumAverage_OrigGray_3_03_256', 'Texture_SumEntropy_OrigGray_3_00_256', 'Texture_SumEntropy_OrigGray_3_01_256', 'Texture_SumEntropy_OrigGray_3_02_256', 'Texture_SumEntropy_OrigGray_3_03_256', 'Texture_SumVariance_OrigGray_3_00_256', 'Texture_SumVariance_OrigGray_3_01_256', 'Texture_SumVariance_OrigGray_3_02_256', 'Texture_SumVariance_OrigGray_3_03_256', 'Texture_Variance_OrigGray_3_00_256', 'Texture_Variance_OrigGray_3_01_256', 'Texture_Variance_OrigGray_3_02_256', 'Texture_Variance_OrigGray_3_03_256', 'Threshold_FinalThreshold_IdentifyPrimaryObjects', 'Threshold_OrigThreshold_IdentifyPrimaryObjects', 'Threshold_SumOfEntropies_IdentifyPrimaryObjects', 'Threshold_WeightedVariance_IdentifyPrimaryObjects', 'Width_Blood_Cells']
    features = [f for f in features_list if 'ExecutionTime' not in f]
    
    # get data
    X = data_df[features]
    y = data_df[target]

    # perform PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    
    # add pca to DF
    data_df[['pca-2d-one', 'pca-2d-two']] = components

    # generate plots
    plot_pca(data_df, target)
    plot_cm(X, y)
    plot_decision_boundary(components, y)


if __name__ == "__main__":
    for target in ['cell_type', 'wbc_category']:
        baseline_plots(target)
