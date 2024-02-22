# Import libraries
import streamlit as st
import requests
import pandas as pd
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import skew

# Define the FastAPI endpoint URL
fastapi_url = "https://customer-segmentation-backend-7emkch5d3q-uc.a.run.app"

def main():
    st.sidebar.title("E-commerce Customer Segmentation App")
    st.sidebar.markdown('''
        ## About
        This application is used to predict e-commerce customer segments. 
        The purpose of this segmentation is to better understand the diverse needs, 
        preferences, and behaviors of different customer groups. 
        **Generative AI** is used to figure out the next marketing strategy based on the segmentation results.
        
        Documentation:
        - [Streamlit](https://streamlit.io/)
        - [Fastapi](https://fastapi.tiangolo.com/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
                 
        ''')
    
    def app():
        st.title('Customer Segment Prediction')
        random_user_id = random.randint(1, 99999)
        user_id = st.number_input("Customer ID:",min_value=1,max_value=99999,value=random_user_id)
        count_orders = st.number_input("Number of Orders:",min_value=1,max_value=999)
        average_spend = st.number_input("Average Spend:",min_value=0.01,max_value=99999.99)
        return_ratio = st.number_input("Return Ratio:",min_value=0.0,max_value=1.0)

        predict = st.button('Predict')
        if predict:
            data = {
                'count_orders': count_orders,
                'average_spend': average_spend,
                'return_ratio': return_ratio
            }
            response = requests.post(fastapi_url +"/predict", json=data)
            # Check if the request was successful
            if response.status_code == 200:
                with st.spinner("Please wait..."):
                    result = response.json()
                    df_prediction = pd.DataFrame({"Segment_ID_Prediction": [result]})
                    info = f"Customer ID:{user_id}, Average Spend: ${average_spend}, Number of Order: {count_orders}, Return ratio: {return_ratio}, Predicted Segment: {df_prediction['Segment_ID_Prediction'].iloc[0]}"
                    cust_info = {
                                "cust_info": info
                    }
                    gen_ai_response = requests.post(fastapi_url+"/ai",json=cust_info)
                    ai_result = gen_ai_response.json()
                    result_string = '\n'.join(ai_result)
                    # gen_ai_response = chain.run(cust_info=cust_info,ai_response=ai_response)
                    st.success(f"### AI Response :\n {result_string}")
        st.write(f"[FastAPI documentation]({fastapi_url}/docs)")
    def eda():
        df = pd.read_csv('ecommerce-cluster.csv')
        df.drop('Unnamed: 0',axis=1,inplace=True)
        df["returned"] = df["status"] == "Returned"

        count_orders = df.groupby(["user_id"])["order_id"].count()
        average_spend = df.groupby(["user_id"])["sale_price"].mean()
        returned = df.groupby(["user_id"])["returned"].sum()
        return_ratio = returned / count_orders

        df_customer = count_orders.rename("count_orders").to_frame()
        df_customer["average_spend"] = round(average_spend,2)
        df_customer["return_ratio"] = return_ratio

        st.header('Exploratory Data Analysis')
        eda_navigation = st.radio("",["EDA","EDA after Clustering"])

        if eda_navigation == "EDA":
            eda_1 = st.selectbox("Select EDA",['Original Dataset','Number of Orders','Average Spend','Returned Orders','Return Ratio','Histogram and Boxplot']) 
            if eda_1 == 'Original Dataset':
                st.write('## Dataset')
                st.write('The dataset used is  [theLook eCommerce public dataset](https://console.cloud.google.com/marketplace/product/bigquery-public-data/thelook-ecommerce).')
                st.dataframe(df)
                st.write(''' The dataset consists of the following fields:

1. `user_id:` This column represents a unique identifier for each user or customer. It is used to associate each order with a specific customer.
2. `order_id:` This column represents a unique identifier for each order. It is used to differentiate between different transactions or purchases made by users.
3. `sale_price:` This column represents the sale price of the product associated with the order. The sale price indicates the amount paid by the customer for the respective product.
4. `created_at:` This column represents the timestamp indicating when the order was created or placed. It provides information about the date and time when the customer initiated the purchase.
5. `status:` This column indicates the current status of the order. The possible values include:
    - "Cancelled": The order was canceled before it was fulfilled.
    - "Shipped": The order has been shipped to the customer.
    - "Complete": The order has been successfully processed and completed.
    - "Processing": The order is currently being processed and has not yet been shipped or completed.
    - "Returned": The order was returned by the customer, indicating a reversal of the sale.      

            ''')
            elif eda_1 == 'Number of Orders':
                # Top 10 users with the Most Orders
                st.header('Number of Order per Customer')                
                top10_users_orders = count_orders.nlargest(10)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                top10_users_orders.plot(kind='barh', ax=ax)
                ax.set_title('Top 10 Users with the Most Orders')
                ax.set_xlabel('Number of Orders')
                ax.set_ylabel('User ID')

                # Display the plot in Streamlit app
                st.pyplot(fig)
                st.write('The visualization and table above represent the number of orders made by different users. User with `user_id` **78119** has the highest number of orders, totaling **14**.')
            
            elif eda_1 == 'Average Spend':
                st.header('Average Spend per Customer')
                # Top 10 users with the most spend
                top10_users_average_spend = average_spend.nlargest(10)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                top10_users_average_spend.plot(kind='barh', ax=ax)
                ax.set_title('Top 10 Users with the Most Average Spend')
                ax.set_xlabel('Average Spend')
                ax.set_ylabel('User ID')

                # Display the plot in Streamlit app
                st.pyplot(fig)
                st.write('The visualization and table above represent the average spend made by different users. Users with `user_id` **33002** have the highest average spending amount of **999.0**.')

            elif eda_1 == 'Returned Orders':
                st.header('Returned Order per Customer')
                # Top 10 Users with the Most Returned Orders
                top10_users_returned = returned.nlargest(10)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(top10_users_returned, labels=top10_users_returned.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Top 10 Users with the Most Returned Orders')

                # Display the plot in Streamlit app
                st.pyplot(fig)
                st.write('The visualization and table above represent the users with the most returned orders. Users with `user_id` **56329** have the most returned orders.')

            elif eda_1 == 'Return Ratio':
                st.header('Return Ratio')
                st.write('Calculating return ratio per customer. The return ratio is the percentage of orders made by a customer that are later returned.')
                # Compiling a conclusive dataframe for the development of a machine learning model for customer segmentation.
                st.dataframe(df_customer)
                st.write("""The `return_ratio` provides insights into the return behavior of customers. A lower `return_ratio` suggests that a user tends to keep their purchases, while a higher `return_ratio` may indicate a greater likelihood of returning items.                       
                         The value return ratio ranging from 0.0 to 1.0. A `return_ratio` of 0.0 means that none of the user's orders were returned (no returns). A `return_ratio` of 1.0 means that all of the user's orders were returned (100% returns).
                         
                         """)

            else:
                st.header('Data Distribution')
                st.write("Understanding and identifying data distribution is crucial in data analysis as they can impact the accuracy of statistical measures and machine learning models. It's important to investigate the nature of these outliers to determine if they are valid data points or if they represent errors or anomalies that need further attention.")
                st.write('---')
                cols = ['count_orders','average_spend','return_ratio']
                for i in cols:
                    st.write(f'### Column: {i}')
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    df_customer[i].plot(kind="hist", bins=100, ax=ax[0])
                    ax[0].set_title(str(i) + " -Histogram")
                    df_customer[i].plot(kind="box", ax=ax[1])
                    ax[1].set_title(str(i) + " -Boxplot")
                    st.pyplot(fig)

                        # Calculate skewness
                    skewness = skew(df_customer[i])

                    # Calculate IQR
                    Q1 = df_customer[i].quantile(0.25)
                    Q3 = df_customer[i].quantile(0.75)
                    IQR = Q3 - Q1

                    # Count outliers
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df_customer[(df_customer[i] < lower_bound) | (df_customer[i] > upper_bound)][i]

                    # Print the number of outliers and percentage
                    num_outliers = len(outliers)
                    total_values = len(df_customer[i])
                    percentage_outliers = (num_outliers / total_values) * 100

                    st.write("Q1: ",Q1)
                    st.write("Q3: ",Q3)
                    st.write(f"Number of outliers in {i}: {len(outliers)}")
                    st.write(f"Percentage of outliers in {i}: {percentage_outliers:.2f}%")
                    st.write(f"Skewness of {i}: {skewness:.2f}")
                    st.write("---")

        else:
            eda_2 = st.selectbox("Select EDA",['Customer per Segment','Relationship between Variable','Characteristics for each segment'])
            segments_pd = pd.read_csv('df_segment.csv')
            segments_pd.drop('user_id',axis=1,inplace=True)
            if eda_2 == 'Customer per Segment':
                st.header('Number of customer per Segment')

                # Count the occurrences of each segment_id
                counts = segments_pd['segment_id'].value_counts().sort_index()

                # Choose a color for each bar
                colors = ['blue', 'green', 'orange', 'red', 'purple']
                # Plot the bar chart using Streamlit
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(counts.index, counts, color=colors)
                ax.set_title('Counts of Each Segment')
                ax.set_xlabel('Segment ID')
                ax.set_ylabel('Count')

                # Display the plot in Streamlit app
                st.pyplot(fig)
                st.write('It can be seen from the visualization above that segment ID 4 has the highest number of customers.')
            elif eda_2 == 'Relationship between Variable':
                st.header('Relationship between pairs of variables for each segment.')
                # Create a Seaborn pairplot
                sns.set(style="ticks")
                pairplot = sns.pairplot(segments_pd, hue='segment_id', palette='Set1')
                pairplot.fig.suptitle('Pairplot of Count Orders, Average Spend, Return Ratio by Segment', y=1.02)

                # Display the pairplot in Streamlit app
                st.pyplot(pairplot)
                st.write('')
                # Create a 3D scatter plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')

                # Scatter plot for each segment
                for segment_id, segment_data in segments_pd.groupby('segment_id'):
                    ax.scatter(segment_data['count_orders'], segment_data['average_spend'], segment_data['return_ratio'], label=f'Segment {segment_id}')

                # Set labels for each axis
                ax.set_xlabel('Count Orders')
                ax.set_ylabel('Average Spend ($)')
                ax.set_zlabel('Return Ratio')

                # Set the title of the plot
                ax.set_title('3D Scatter Plot of Segments')

                # Display the legend
                ax.legend()

                # Display the plot in Streamlit app
                st.pyplot(fig)
            else:
                st.header("Characteristics for each segment")
                st.write('## Average Value per segment')
                # Set 'segment_id' as the index and calculate the average for each column
                averages_per_segment = segments_pd.groupby('segment_id').mean()
                averages_per_segment.rename(columns={'count_orders': 'average_orders_count','return_ratio':'average_return_ratio'}, inplace=True)
                st.dataframe(averages_per_segment)

                # Get the number of columns in your DataFrame
                num_columns = len(averages_per_segment.columns)

                # Define a color dictionary mapping segment_id to a color
                color_dict = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'purple'}

                # Create subplots with a number of rows based on the number of columns
                fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(15, 5))

                # Iterate over each column and plot a bar chart
                for i, (column, color) in enumerate(zip(averages_per_segment.columns, sns.color_palette("husl", len(color_dict)))):
                    ax = axes[i] if num_columns > 1 else axes  # Use the same axis if there's only one column
                    for segment_id, segment_color in color_dict.items():
                        subset = averages_per_segment.loc[averages_per_segment.index == segment_id]
                        ax.bar(segment_id, subset[column].values, color=segment_color, label=f'Segment {segment_id}')

                    ax.set_title(column)
                    ax.set_xlabel('Segment ID')
                    ax.set_ylabel('Average Value')
                    ax.legend()

                # Adjust layout
                plt.tight_layout()

                # Display the plot in Streamlit app
                st.pyplot(fig)
                
                st.write('## Minimum and Maximum Value per segment')
                # Set 'segment_id' as the index and calculate the minimum for each column
                minimum_per_segment = segments_pd.groupby('segment_id').min()
                minimum_per_segment.rename(columns={'count_orders': 'minimum_orders_count','average_spend':'minimum_average_spend','return_ratio':'minimum_return_ratio'}, inplace=True)
                
                # Set 'segment_id' as the index and calculate the maximum for each column
                maximum_per_segment = segments_pd.groupby('segment_id').max()
                maximum_per_segment.rename(columns={'count_orders': 'maximum_orders_count','average_spend':'maximum_average_spend','return_ratio':'maximum_return_ratio'}, inplace=True)

                combined_table = pd.concat([averages_per_segment,minimum_per_segment, maximum_per_segment], axis=1)
                combined_table = combined_table[['minimum_orders_count','maximum_orders_count','average_orders_count','minimum_average_spend','maximum_average_spend','average_spend','minimum_return_ratio','maximum_return_ratio','average_return_ratio']]
                st.dataframe(combined_table)

                st.write('## Violin Plots by Segment')

                # Define color palettes for each segment
                segment_colors = sns.color_palette("husl", n_colors=len(segments_pd['segment_id'].unique()))

                # Sort unique segment IDs
                unique_segments_sorted = sorted(segments_pd['segment_id'].unique())

                # Loop through each segment and column to create violinplots
                for i, segment_id in enumerate(unique_segments_sorted):
                    plt.figure(figsize=(12, 8))

                    for j, column_name in enumerate(segments_pd.columns[:-1], start=1):  # Exclude 'user_id'
                        subset_segments_pd = segments_pd[segments_pd['segment_id'] == segment_id]

                        plt.subplot(1, 3, j)
                        sns.violinplot(y=subset_segments_pd.loc[:, column_name], color=segment_colors[i])
                        plt.title(f"Segment {segment_id} - {column_name} Violin Plot")
                        print('\n')

                    plt.tight_layout()
                    
                    # Display the violin plot in Streamlit app
                    st.pyplot(plt)

                st.write(f'''
                         - Segment 1: Spend between 52.75 and 112.75 with Average spend 74.31, Number of order per person is between: 1.00 order and 12.00 orders with average order: 2.27, Return ratio is between 0.00 and 1.00 with average return ratio: 0.10 
- Segment 2: Spend between 112.75 and 233.00 with Average spend 150.89, Number of order per person is between: 1.00 order and 10.00 orders with average order: 1.67, Return ratio is between 0.00 and 1.00 with average return ratio: 0.10 
- Segment 3: Spend between 558.75 and 999.00 with Average spend 801.13, Number of order per person is between: 1.00 order and 2.00 orders with average order: 1.01, Return ratio is between 0.00 and 1.00 with average return ratio: 0.10 
- Segment 4: Spend between 0.02 and 52.78 with Average spend 31.24, Number of order per person is between: 1.00 order and 14.00 orders with average order: 1.85, Return ratio is between 0.00 and 1.00 with average return ratio: 0.10 
- Segment 5: Spend between 233.66 and 550.00 with Average spend 314.91, Number of order per person is between: 1.00 order and 9.00 orders with average order: 1.44, Return ratio is between 0.00 and 1.00 with average return ratio: 0.10
                         ''')

    # Radio button to choose between manual or chatbot:
    navigation = st.sidebar.radio("Page Navigation", ["Prediction 🖱️", "EDA 📊"])

    # Conditionally display the appropriate prediction form
    if navigation == "Prediction 🖱️":
        app()
    elif navigation == "EDA 📊":
        eda()

    st.sidebar.markdown(''' 
    ## Created by: 
    Ahmad Luay Adnani - [GitHub](https://github.com/ahmadluay9) 
    ''')

if __name__ == '__main__':
    main()