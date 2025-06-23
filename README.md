# PolIce force bulgary assistance - Fighting to reduce residential burglary
TU/e Multidisciplinary CBL, helping London police force reduce burlgary rates by predicting crime, allocating their police officers and adding our original twist by including the community in our innovative solution.

This README will give the information necessary to run the project and a summary of its content. This repository was created due to many members having issues in the git history after pushing big data files. This is the second repository used in the project, avoiding the previous problem using a detailed gitignore file. To access the previous repository to see previous group history you can view: 
-> https://github.com/ala-mn/Addressing-real-world-crime-and-security-problems-with-data-science.git

## Our GOAL
The aim of this project was a vague problem to which we had to find a solution, how can we, as a team of Data Scientists and Computer Scientists help the London police force in reducing burglary rates. 

The most obvious solution was predicting burglaries for future month(s). However, we were not satisfied by this solution, and considered that a big part of the police force is not only to reduce the numerical value of burglary rates, but also to make the community feel safe and protected by them, this is why we have developed a solution that not only predicts burglaries with up to 80% R^2, but also takes input from the London community and uses it to give police more in-depth insights. 

## Project Structure

community-tool/            # Front-end for community survey and visualizations<br>
  ├─ index.html            # Main html structure of community website<br>
  ├─ feedback.html         # Community survey page<br>
  ├─ dashboard.html        # Community visualizations dashboard<br>
  ├─ visuals.js            # JS for community visualizations<br>
  ├─ script.js             # Interaction logic for survey<br>
  └─ style.css             # Styles for community tool<br>

data/                      # Input and output data files<br>
  ├─ burglary_next_month_forecast.csv   # Model outputs (predicted burglaries)<br>
  ├─ crime_fixed_data.csv               # Master historical burglary dataset<br>
  ├─ topic_sentiment_summary.csv        # Processed community feedback by topic & sentiment<br>
  ├─ LSOAs.geojson                      # Boundaries for LSOA polygon maps<br>
  ├─ wards.geojson                      # Boundaries for London wards<br>
  └─ ...                                # Other CSVs (population, stop-and-search, IMD, etc.)<br>

models/                    # Model artifacts and training code<br>
  ├─ xgb_burglary_model.pkl             # Trained XGBoost model<br>
  ├─ robust_scaler.pkl                  # Scaler for feature preprocessing<br>

Police-dashboard/         # Dash app for police users<br>
  ├─ app.py                     # Main Dash application code<br>
  ├─ helper.py                  # Utilities (prediction saving, spatial joins)<br>
  ├─ process_data.py            # Upload data function process file<br>

Other files                     # Normalization, data exploration, etc.<br>
