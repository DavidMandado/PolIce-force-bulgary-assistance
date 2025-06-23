# PolIce force bulgary assistance - Fighting to reduce residential burglary
TU/e Multidisciplinary CBL, helping London police force reduce burlgary rates by predicting crime, allocating their police officers and adding our original twist by including the community in our innovative solution.

This README will give the information necessary to run the project and a summary of its content. This repository was created due to many members having issues in the git history after pushing big data files. This is the second repository used in the project, avoiding the previous problem using a detailed gitignore file. To access the previous repository to see previous group history you can view: 
-> https://github.com/ala-mn/Addressing-real-world-crime-and-security-problems-with-data-science.git

## Our GOAL
The aim of this project was a vague problem to which we had to find a solution, how can we, as a team of Data Scientists and Computer Scientists help the London police force in reducing burglary rates. 

The most obvious solution was predicting burglaries for future month(s). However, we were not satisfied by this solution, and considered that a big part of the police force is not only to reduce the numerical value of burglary rates, but also to make the community feel safe and protected by them, this is why we have developed a solution that not only predicts burglaries with up to 80% R^2, but also takes input from the London community and uses it to give police more in-depth insights. 

## Project Structure

community-tool/            # Front-end for community survey and visualizations
  ├─ index.html            # Main html structure of community website
  ├─ feedback.html         # Community survey page
  ├─ dashboard.html        # Community visualizations dashboard
  ├─ visuals.js            # JS for community visualizations
  ├─ script.js             # Interaction logic for survey
  └─ style.css             # Styles for community tool

data/                      # Input and output data files
  ├─ burglary_next_month_forecast.csv   # Model outputs (predicted burglaries)
  ├─ crime_fixed_data.csv               # Master historical burglary dataset
  ├─ topic_sentiment_summary.csv        # Processed community feedback by topic & sentiment
  ├─ LSOAs.geojson                      # Boundaries for LSOA polygon maps
  ├─ wards.geojson                      # Boundaries for London wards
  └─ ...                                # Other CSVs (population, stop-and-search, IMD, etc.)

models/                    # Model artifacts and training code
  ├─ xgb_burglary_model.pkl             # Trained XGBoost model
  ├─ robust_scaler.pkl                  # Scaler for feature preprocessing

Police-dashboard/         # Dash app for police users
  ├─ app.py                     # Main Dash application code
  ├─ helper.py                  # Utilities (prediction saving, spatial joins)
  ├─ process_data.py            # Upload data function process file

Other files                     # Normalization, data exploration, etc.
