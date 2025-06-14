SpaceX Falcon 9 First Stage Landing Prediction
🚀 Project Overview

This repository contains the capstone project for the IBM Data Science Professional Certificate on Coursera. The project focuses on applying the full data science lifecycle to predict the landing success of the SpaceX Falcon 9 first stage. By leveraging various data collection techniques, performing exploratory data analysis, creating interactive visualizations, and building a predictive machine learning model, this project demonstrates a comprehensive, hands-on approach to solving a real-world data science problem.

The ultimate goal is to determine if the first stage will land successfully, which is a critical factor in reducing the cost of space launches, making them more accessible and frequent.
🎯 Project Objectives

    Data Collection: Gather historical launch data from the SpaceX REST API and through web scraping from Wikipedia.

    Data Wrangling & Cleaning: Preprocess and clean the collected data to prepare it for analysis.

    Exploratory Data Analysis (EDA): Analyze the data using SQL and data visualization techniques to uncover patterns and insights related to launch success.

    Interactive Dashboard: Develop an interactive dashboard using Plotly Dash to visualize launch site locations, success rates, and the relationship between launch parameters and outcomes.

    Machine Learning: Build and evaluate a classification model to predict whether the Falcon 9 first stage will land successfully.

🛠️ Technologies & Libraries

This project utilizes a suite of Python libraries and data science tools:

    Data Manipulation & Analysis: Pandas, NumPy

    Data Collection: requests library (for API), BeautifulSoup (for web scraping)

    Database: sqlite3 (for SQL queries)

    Data Visualization: Matplotlib, Seaborn, Plotly

    Interactive Dashboard: Dash

    Machine Learning: Scikit-learn

    Development Environment: Jupyter Notebook

📂 Repository Structure

.
├── LICENSE
├── README.md
├── SpaceX - Module 1-1 - Data Collection API Lab.ipynb
├── SpaceX - Module 1-2 - Data Collection with Web Scraping lab.ipynb
├── SpaceX - Module 1-3 - Data Wrangling.ipynb
├── SpaceX - Module 2 - EDA with SQL.ipynb
├── SpaceX - Module 2 - EDA with Visualization Lab.ipynb
├── SpaceX - Module 3-1 - Interactive Visual Analytics.ipynb
└── SpaceX - Module 4 - Machine-Learning Prediction.ipynb

File Descriptions:

    SpaceX - Module 1-1 - Data Collection API Lab.ipynb: Collects data from the SpaceX v4 API.

    SpaceX - Module 1-2 - Data Collection with Web Scraping lab.ipynb: Scrapes and parses launch data from a Wikipedia table.

    SpaceX - Module 1-3 - Data Wrangling.ipynb: Cleans, merges, and prepares the datasets for analysis.

    SpaceX - Module 2 - EDA with SQL.ipynb: Performs initial exploratory analysis using SQL queries on the cleaned dataset.

    SpaceX - Module 2 - EDA with Visualization Lab.ipynb: Uses Matplotlib and Seaborn to visualize relationships within the data.

    SpaceX - Module 3-1 - Interactive Visual Analytics.ipynb: Creates interactive charts with Plotly and builds a full dashboard application with Dash.

    SpaceX - Module 4 - Machine-Learning Prediction.ipynb: Develops, trains, and evaluates various classification models to predict landing success.

    LICENSE: Project license details.

    README.md: This file.

📈 Methodology

The project follows a structured data science pipeline:

    Data Collection:

        Fetched launch data attributes (flight number, payload, launch site, etc.) from the SpaceX API.

        Scraped a Wikipedia page for a table of Falcon 9 launches to supplement the API data.

    Data Wrangling:

        Handled missing values and standardized data formats across the two data sources.

        Created a unified, clean dataset ready for analysis.

        Performed feature engineering by one-hot encoding categorical variables like LaunchSite and LandingPad.

    Exploratory Data Analysis (EDA):

        Used SQL to perform initial queries and aggregations to understand the dataset's basic characteristics.

        Visualized key features like launch success rates per launch site, the relationship between payload mass and success, and success trends over time.

    Interactive Dashboard:

        Developed a Plotly Dash dashboard featuring:

            A dropdown to filter launch success statistics by launch site.

            A scatter plot showing the relationship between payload mass and landing outcome, filterable by launch site.

            An interactive map showing launch sites.

    Machine Learning Modeling:

        Split the data into training and testing sets.

        Scaled the features using StandardScaler.

        Trained and evaluated several classification algorithms, including:

            Logistic Regression

            Support Vector Machine (SVM)

            Decision Tree

            K-Nearest Neighbors (KNN)

        Used Grid Search Cross-Validation (GridSearchCV) to find the best hyperparameters for the top-performing model.

        Evaluated the final model's performance using metrics like accuracy and a confusion matrix.

📊 Key Results

    The analysis revealed a strong correlation between the launch site, orbit type, and the probability of a successful landing.

    The interactive dashboard provides an intuitive way to explore how different factors, such as payload mass, affect landing success for each launch site.

    The final machine learning model achieved a high accuracy score in predicting the landing outcome, demonstrating the feasibility of using historical data to forecast future mission success. The Support Vector Machine (SVM) was identified as the best-performing model after hyperparameter tuning.

🚀 How to Use

To run this project locally, follow these steps:

    Clone the repository:

    git clone https://github.com/medelassal/Coursera-Applied-Data-Science-Capstone-SpaceX-Launch-Analysis.git
    cd Coursera-Applied-Data-Science-Capstone-SpaceX-Launch-Analysis

    Install the required libraries:

    pip install -r requirements.txt

    (Note: You may need to create a requirements.txt file from the notebooks' imports if one is not provided.)

    Run the Jupyter Notebooks:
    Launch Jupyter Notebook and open the .ipynb files in the recommended order (from Module 1 to 4) to follow the project workflow.

    jupyter notebook

📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.
