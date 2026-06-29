# Quantification of the Tactical impact of **man-marking** and isolating **playmakers** through a Network Resilience Analysis. 

This repository contains the data science framework developed for the **Sports Data Science Final Assignment**, utilizing the historical Wyscout datasets from the 2017-2018 season provided by Pappalardo et al. 

The core of the project is built around an interactive pipeline (`app.py`) that allows the user to select a team and a specific playmaker. The framework automatically extracts the data, runs the network resilience simulations, and generates a raw HTML report template. The user can then add their personal qualitative football analyst interpretations directly into the pipeline to compile the final tactical report.

Currently, the `report/` folder contains two finalized and very opposite examples of these network resilience outputs, complete with my personal  match analyst insights on Éver Banega (Sevilla FC) and Lucas Biglia (AC Milan) impacts on their respective teams.
---

## 📂 Project Structure

```text

├── data/
│   ├── Code/
│   │   └── soccer_nsd_code.ipynb       # Reference explanatory Jupyter Notebook (Pappalardo et al.)

│   ├── Data/                           # [GitIgnored] Raw JSON files
│   │   ├── events/                     
│   │   ├── matches/                    
│   │   ├── coaches.json 
│   │   ├── competitions.json 
│   │   ├── eventid2name.csv 
│   │   ├── playerrank.json 
│   │   ├── players.json 
│   │   ├── referees.json 
│   │   ├── tags2name.csv 
│   │   └── teams.json 
 
│   ├── data_paper_soccer_nsd.pdf       # Reference data paper (Pappalardo et al.) 
│   ├── PlayerRank_paper.pdf            # Reference methodology paper (Pappalardo et al.) 
│   └── readme.txt                      # Data documentation (Pappalardo et al.) 
 

├── report/ 
│   ├── report_AC_Milan_L._Biglia.html  # Personal tactical evaluation report for AC Milan case study  
│   └── report_Sevilla_FC_É._Banega.html # Personal tactical evaluation report for Sevilla FC case study  
 
├── src/ 
│   ├── functions.py                    # All the functions used in to develop app.py  
 
│   └── visuals.py                      # Graphs creation functions  
 
├── .gitignore                          # Git ignore file  
 
├── app.py                              # MAIN SCRIPT EXECUTION PIPELINE generating HTML output files 

└── README.md                           # Project documentation (this file)
```

## 🚀 How to Run the Project

1. Prerequisites

Ensure you have Python 3.12.4 installed along with the required libraries imported in the various files. Then download Pappalardo's event data JSON files and place them as it's presented in 'Project Structure'

2. Core Architecture & Pipeline (`app.py`)

The entire analysis workflow is contained in `app.py` and divided into 5 cells to gradually run in an interactive window.

These will sequentially import the required packages, let the user select a team, a playmaker from the team after inspecting their passing total weights, a starting 11 around the specific player and for last it will generate a final HTML output file containing different metrics and visualizations to be interpreted.

3. Analytics & Metrics Computed


## 👥 Author
_Livio Guerra - Leiden University (Student ID: s4444159)_ 
_Email: l.guerra@umail.leidenuniv.nl_