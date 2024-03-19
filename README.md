# Generative AI for Personalized E-commerce Customer Segmentation
- [FastAPI Documentation](https://customer-segmentation-backend-ewh36wojna-uc.a.run.app/docs)
- [App Demo](https://gen-ai-cust-frontend.streamlit.app/)

This project aims to utilize Generative AI for the next marketing strategy in the case of e-commerce customer segmentation.

https://github.com/ahmadluay9/Generative-AI-for-Personalized-E-commerce-Customer-Segmentation/assets/123846438/a94cce23-45b0-4ebd-9706-09fe4310cfb8

## File explanation

This repository consists of several files :
```
    ┌── Backend/
    │   ├── model/
    │   |   ├── model.pkl
    │   |   ├── model.py
    │   ├── ai.py
    │   ├── ai_response.txt
    │   ├── app.py
    │   ├── dockerfile
    │   ├── requirements.txt
    ├── Frontend/
    │   ├── app.py
    │   ├── df_customer.csv
    │   ├── df_segment.csv
    │   ├── dockerfile
    │   ├── ecommerce-cluster.csv
    │   ├── requirements.txt
    ├── docker-compose.yml
    ├── notebook.ipynb
    └── README.md 
```

- `backend/ model/ model.pkl`: This file is a trained machine learning model for clustering case.
  
- `backend/ model/ model.py`: This file contains code to load the trained machine learning model from the saved file.

- `backend/ ai.py`: This file contains the backend code for generative ai.

- `backend/ ai_respones.txt`: This file contains saved AI response for each customer segment.

- `backend/ app.py`: This file contains the backend code for the application. It responsible for handling server-side logic, API endpoints, or any other backend functionality.

- `backend/ dockerfile`: Dockerfile is used to build a Docker image for backend application. It includes instructions on how to set up the environment and dependencies needed for backend.

- `backend/ requirements.txt`: This file lists the Python dependencies required for backend application. These dependencies can be installed using a package manager like pip.

- `frontend/ app.py`: This file is the main script for the frontend of the application and is developed using the Streamlit framework. It contain sections for user input, and the integration of backend functionality through API calls.

- `frontend/ df_customer.csv`: This CSV file is the result of exploratory data analysis and is used to train the model for clustering.

- `frontend/ df_segment.csv`: This CSV file is the result of clustering.

- `frontend/ dockerfile`: Similar to the backend Dockerfile, this file is used to build a Docker image for frontend application. It includes instructions on setting up the environment and installing dependencies.

-  `frontend/ ecommerce-cluster.csv`: This CSV file is the result of a query from Google BigQuery.

- `frontend/ requirements.txt`: This file lists the Python dependencies required for frontend application. These dependencies can be installed using a package manager like pip. 

- `docker-compose.yml` : This is a configuration file for Docker Compose. It defines services, networks, and volumes for your application's containers. Docker Compose simplifies the process of running multi-container applications.

- `README.md`: This is a Markdown file that typically contains documentation for the project. It include information on how to set up and run your application, dependencies, and any other relevant details.

- `notebook.ipynb`: This Jupyter Notebook file contain code, analysis, or documentation related to machine learning tasks using Google Cloud's Vertex AI.

---

## Brief Summary of this Project

The flow of this project starts with Exploratory Data Analysis (EDA) to understand the basic structure of the dataset. Next, we determine the number of segments based on the distortion score elbow and silhouette score. Following that, we train the model and make predictions using K-Means Clustering. The clustering results reveal 5 customer segments with specific characteristics. Generative AI is then employed to determine the marketing strategy steps for each customer segment.

## Project Conclusion
After conducting Exploratory Data Analysis (EDA) and segmenting customers into five distinct groups based on spending patterns, order frequency, and return ratios, Generative AI was employed to devise tailored marketing strategies for each segment.

- Segment 1: Customers with moderate spending (52.75 to 112.75), averaging $74.31, and an average of 2.27 orders per person. A suggested marketing strategy is to offer personalized recommendations to encourage repeat purchases.

- Segment 2: Customers with higher spending (112.75 to 233.00), averaging $150.89, and an average of 1.67 orders per person. A suggested marketing strategy is to introduce a loyalty program to reward repeat purchases.

- Segment 3: High-spending customers (558.75 to 999.00), averaging $801.13, with an average of 1.01 orders per person. A suggested marketing strategy is to offer exclusive, high-end products or services to enhance the premium shopping experience.

- Segment 4: Lower-spending customers (0.02 to 52.78), averaging $31.24, and an average of 1.85 orders per person. A suggested marketing strategy is to introduce subscription services or bundle deals to increase customer retention.

- Segment 5: Customers with significant spending (233.66 to 550.00), averaging $314.91, and an average of 1.44 orders per person. A suggested marketing strategy is to create exclusive VIP events or experiences to appreciate and retain high-value customers.

