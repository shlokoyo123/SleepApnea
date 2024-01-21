import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from sklearn import preprocessing
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
pd.DataFrame.iteritems = pd.DataFrame.items #https://stackoverflow.com/questions/76404811/attributeerror-dataframe-object-has-no-attribute-iteritems


def dictConverter(df, col1, col2):
    tempDF = df[[col1, col2]].drop_duplicates()
    return dict(zip(tempDF[col1], tempDF[col2]))
def EDA():
    # Reading data file
    df = pd.read_csv("Data\\SleepApnea.csv")

    with open("output\\output.txt", 'w') as f:
        # Reviewing dataset
        f.write("Reviewing Dataset: \n Head of dataset: \n")
        f.write(tabulate(df.head(5), df.columns, tablefmt="psql"))
        f.write("\n Tail of dataset: \n")
        f.write(tabulate(df.tail(5), df.columns, tablefmt="psql"))
        f.write("\nMore description about dataset: \n")
        f.write(str(df.info()))
        f.write("\n")
        f.write(str(df.describe()))
        f.write("\n")
        f.write("Exploring categorical variables: \n")
        f.write("\n")
        f.write("Unique Values of Occupation are: \n\t")
        f.write(str(df['Occupation'].unique()))
        f.write("\n")
        f.write("\nUnique Values of BMI Category are: \n\t")
        f.write(str(df['BMI Category'].unique()))
        f.write("\n")
        f.write("\nUnique Values of Sleep Disorder are: \n\t")
        f.write(str(df['Sleep Disorder'].unique()))
    f.close()

    #1. Expanding Blood Pressure values into systolic(higher) and diastolic(lower) values
    df1 = (pd.concat([df, df['Blood Pressure'].str
                        .split('/', expand=True)], axis=1)
                        .drop('Blood Pressure', axis=1))
    df1 = df1.rename(columns={0: 'Systolic (Blood Pressure)', 1: 'Diastolic (Blood Pressure)'})
    df1['Systolic (Blood Pressure)'] = df1['Systolic (Blood Pressure)'].astype(float)
    df1['Diastolic (Blood Pressure)'] = df1['Diastolic (Blood Pressure)'].astype(float)

    #Opening file in an append mode
    with open('output\\output.txt', 'a') as f:
        f.write("\nData frame info :\n")
        f.write(str(df1.info()))
    f.close()

    #2. Encoding categorical values into integral values
    label_encoder = preprocessing.LabelEncoder()
    df1['GenderIndex'] = label_encoder.fit_transform(df1['Gender'])
    df1['OccupationIndex'] = label_encoder.fit_transform(df1['Occupation'])
    df1['BMI CategoryIndex'] = label_encoder.fit_transform(df1['BMI Category'])
    df1['Sleep DisorderIndex'] = label_encoder.fit_transform(df1['Sleep Disorder'])

    # Creating a dictionary
    genderDict = dictConverter(df1, 'Gender', 'GenderIndex')
    occupationDict = dictConverter(df1, 'Occupation', 'OccupationIndex')
    bmiDict = dictConverter(df1, 'BMI Category', 'BMI CategoryIndex')
    sleepDisorderDict = dictConverter(df1, 'Sleep Disorder', 'Sleep DisorderIndex')

    with open('output\\output.txt', 'a') as f:
        f.write("\nReference to categorical dictionaries:\n\tGender Dict: \n")
        f.write(str(genderDict))
        f.write("\n\tOccupation Dict: \n")
        f.write(str(occupationDict))
        f.write("\n\tBMI Dict: \n")
        f.write(str(bmiDict))
        f.write("\n\tSleep Disorder Dict: \n")
        f.write(str(sleepDisorderDict))
    f.close()

    df1.drop(columns = ['Gender', 'Occupation', 'BMI Category' , 'Sleep Disorder'], axis =1, inplace=True)
    df1.rename(columns = {'GenderIndex':'Gender'
                            ,'OccupationIndex':'Occupation'
                                , 'BMI CategoryIndex':'BMI Category'
                                    , 'Sleep DisorderIndex':'Sleep Disorder'
                          }, inplace=True)
    #df1.head()

    #3. Outlier removal
    num_col = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level','Stress Level','Heart Rate', 'Daily Steps','Systolic (Blood Pressure)', 'Diastolic (Blood Pressure)']
    Q1 = df1[num_col].quantile(0.25) # Determination of lower quartile
    Q3 = df1[num_col].quantile(0.75) # Determination of upper quartile
    IQR = Q3 - Q1                    #Interquartile range

    # Filtering out the outlier values
    df1 = df1[~((df1[num_col] < (Q1 - 1.5 * IQR)) | (df1[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)]
    with open('output\\output.txt', 'a') as f:
        f.write("\nReviewing of dataframe df1: \n")
        f.write(tabulate(df1.head(5), df1.columns, tablefmt="psql"))
    f.close()
    df1.drop(columns='Person ID', axis=1).corr()



    # Generate Plots
    # Correlation Heatmap
    """
    fig = px.imshow(df1.drop(columns='Person ID', axis=1).corr())
    fig.update_layout(title = {'text': 'Correlation Heatmap',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               }
                      )
    fig.write_image('images/CorrelationHeatmap.png')
    """
    corre = df1.drop(columns='Person ID', axis=1).corr()
    print(corre)
    # Create a correlation matrix
    plt.figure(figsize=(16, 6))
    fig = sns.heatmap(df1.drop(columns='Person ID', axis=1).corr())
    fig.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
    # pio.write_image(fig, 'images/CorrelationHeatmap.png')
    plt.savefig('images/CorrelationHeatmap.png', dpi=300, bbox_inches='tight')

    # Pairplot
    #fig = px.scatter_matrix(df1.drop(['Person ID'], axis=1), color='Sleep Disorder')
    #fig.write_image("images/PairPlot.png")
    plt.figure(figsize=(160, 160))
    sns.set(style="ticks")
    pair_plot = sns.pairplot(df1.drop(['Person ID'], axis=1), hue='Sleep Disorder')
    # Save the figure using Seaborn's savefig
    plt.savefig("images/PairPlot.png", dpi=300, bbox_inches='tight')

    # Histogram by Sleep Disorder
    """
    fig = px.histogram(df1, x='Sleep Duration', color='Sleep Disorder', marginal='rug', nbins=30)
    fig.update_layout(title={'text': 'Histogram by Sleep Disorder',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top',  # new
      },
                      xaxis=dict(title='Sleep Duration'),
                      yaxis=dict(title='Count'),
                      legend=dict(title='Sleep Disorder'),
                      showlegend=True)
    fig.write_image("images/HistogramSleepDisorder.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    hist_plot = sns.histplot(df1, x='Sleep Duration', hue='Sleep Disorder', element='step', stat='count',
                             common_norm=False, bins=30, palette='Set1')
    # Set plot titles and labels
    plt.title('Histogram by Sleep Disorder')
    plt.xlabel('Sleep Duration')
    plt.ylabel('Count')

    # Show the legend
    plt.legend(title='Sleep Disorder')

    # Save the figure using Matplotlib's savefig
    plt.savefig("images/HistogramSleepDisorder.png")
    # Histogram by BMI Category
    """
    fig = px.histogram(df1, x='Sleep Duration', color='BMI Category', marginal='rug', nbins=30)
    fig.update_layout(title={'text': 'Histogram by BMI Category',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               },
                      xaxis=dict(title='Sleep Duration'),
                      yaxis=dict(title='Count'),
                      legend=dict(title='BMI Category'),
                      showlegend=True)
    fig.write_image("images/HistogramBMICategory.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    hist_plot = sns.histplot(df1, x='BMI Category', hue='Sleep Disorder', element='step', stat='count',
                             common_norm=False, bins=30)
    # Set plot titles and labels
    plt.title('Histogram by Sleep Disorder')
    plt.xlabel('BMI Category')
    plt.ylabel('Count')

    # Show the legend
    plt.legend(title='BMI Category')

    # Save the figure using Matplotlib's savefig
    plt.savefig("images/HistogramBMICategory.png")
    # Boxplot by Gender
    """
    fig = px.box(df1, x='Gender', y='Sleep Duration', color='Gender')
    fig.update_layout(title={'text': 'Boxplot by Gender',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               },
                      xaxis=dict(title='Gender'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/HistogramGender.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='Gender', y='Sleep Duration', hue='Gender', palette='Set1')
    # Set plot titles and labels
    plt.title('Boxplot by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(title='Gender')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxplotGender.png")
    # Boxplot by Occupation
    """
    fig = px.box(df1, x='Occupation', y='Sleep Duration', color='Occupation')
    fig.update_layout(title={'text': 'Boxplot by Occupation',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               },
                      xaxis=dict(title='Occupation'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/HistogramOccupation.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='Occupation', y='Sleep Duration', hue='Occupation', palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot by Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(title='Occupation')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxplotOccupation.png")
    # Boxplot by BMI Category
    """
    fig = px.box(df1, x='BMI Category', y='Sleep Duration', color='BMI Category')
    fig.update_layout(title={'text': 'Boxplot by BMI Category',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               },
                      xaxis=dict(title='BMI Category'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/BoxPlotBMICategory.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='BMI Category', y='Sleep Duration', hue='BMI Category', palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot by BMI Category')
    plt.xlabel('BMI Category')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(title='BMI Category')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxplotBMICategory.png")
    # Boxplot by Sleep Disorder

    """
    fig = px.box(df1, x='Sleep Disorder', y='Sleep Duration', color='Sleep Disorder')
    fig.update_layout(title={'text': 'Boxplot by Sleep Disorder',
                               'y': 0.925,  # new
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'  # new
                               },
                      xaxis=dict(title='Sleep Disorder'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/BoxPlotSleepDisorder.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='Sleep Disorder', y='Sleep Duration', hue='Sleep Disorder'
                           , palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot by Sleep Disorder')
    plt.xlabel('Sleep Disorder')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(loc='upper left',title='Sleep Disorder')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxplotSleepDisorder.png")

    # Analysis - "Relationship between sleep duration and body mass index depends on age"

    # Scatterplot with Age, Sleep Duration and BMI Category
    """
    fig = px.scatter(df1, x='Age', y='Sleep Duration', color='BMI Category', hover_data=['Age', 'Sleep Duration'])
    fig.update_layout(title={'text': 'Scatterplot: Age vs Sleep Duration (Color: BMI Category)',
                             'y': 0.925,  # new
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'  # new
                             },
                      xaxis=dict(title='Age'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/ScatterPlot.png")
    """
    # Create a scatter plot using Seaborn
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    scatter_plot = sns.scatterplot(data=df1, x='Age', y='Sleep Duration', hue='BMI Category', palette='viridis',
                                   size='Age')
    # Set plot titles and labels
    plt.title('Scatterplot: Age vs Sleep Duration (Color: BMI Category)')
    plt.xlabel('Age')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(title='BMI Category')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/ScatterPlot.png")

    # Create age group 20s, 30s, 40s, and 50s
    df1['Age_bin'] = pd.cut(df1['Age'], [20, 30, 40, 50, 60], labels=['20s', '30s', '40s', '50s'])

    # Boxplot: BMI Category by Age_bin
    """
    fig = px.box(df1, x='Age_bin', y='BMI Category', color='Age_bin')
    fig.update_layout(title={'text': 'Boxplot: BMI Category by Age_bin',
                             'y': 0.925,  # new
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'  # new
                             },
                      xaxis=dict(title='Age_bin'),
                      yaxis=dict(title='BMI Category'))
    fig.write_image("images/BoxPlotAgeBinBMI.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='Age_bin', y='BMI Category', hue='Age_bin', palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot: BMI Category by Age bin')
    plt.xlabel('Age Bin')
    plt.ylabel('BMI Category')
    # Show the legend
    plt.legend(title='Age Bin')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxPlotAgeBinBMI.png")

    # Boxplot: Sleep Duration by Age_bin
    """
    fig = px.box(df1, x='Age_bin', y='Sleep Duration', color='Age_bin')
    fig.update_layout(title={'text': 'Boxplot: Sleep Duration by Age_bin',
                             'y': 0.925,  # new
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'  # new
                             },
                      xaxis=dict(title='Age_bin'),
                      yaxis=dict(title='Sleep Duration'))
    fig.write_image("images/BoxPlotAgeBinSleepDuration.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df1, x='Age_bin', y='Sleep Duration', hue='Age_bin', palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot: Sleep Duration by Age_bin')
    plt.xlabel('Age Bin')
    plt.ylabel('Sleep Duration')
    # Show the legend
    plt.legend(title='Age Bin')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxPlotAgeBinSleepDuration.png")

    # Age_bin, BMI Category, and Sleep Duration Boxplot by Occupation
    df_long = pd.melt(df1, id_vars=['Occupation']
                      , value_vars=['Age_bin', 'BMI Category', 'Sleep Duration']
                      , var_name='Variable', value_name='Value')

    """
    fig = px.box(df_long, x='Occupation', y='Value', color='Variable')
    fig.update_layout(title={'text': 'Boxplot: Age_bin, BMI Category, and Sleep Duration by Occupation',
                             'y': 0.925,  # new
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'  # new
                             },
                      xaxis=dict(title='Occupation'),
                      yaxis=dict(title='Value'))
    fig.write_image("images/BoxPlotOccupation.png")
    """
    plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
    box_plot = sns.boxplot(data=df_long, x='Occupation', y='Value', hue='Variable', palette='Set2')
    # Set plot titles and labels
    plt.title('Boxplot: Sleep Duration by Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('Value')
    # Show the legend
    plt.legend(title='Value')
    # Save the figure using Matplotlib's savefig
    plt.savefig("images/BoxPlotAgeBinSleepDuration.png")

    with open('output//output.txt', 'a') as f:
        f.write(tabulate(df1.head(5), df1.columns, tablefmt="psql"))
        f.write("\n")
        f.write(str(df1.info()))
    f.close()

    return df1, genderDict, occupationDict, bmiDict, sleepDisorderDict