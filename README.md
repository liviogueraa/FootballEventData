# Quantification of the Tactical impact of **man-marking** and isolating **playmakers** through a Network Resilience Analysis. 

This repository contains the data science framework developed for the **Sports Data Science Final Assignment**, utilizing the historical Wyscout datasets from the 2017-2018 season provided by Pappalardo et al. 

The core of the project is built around an interactive pipeline (`app.py`) that allows the user to select a team and a specific playmaker. The framework automatically extracts the data, runs the network resilience simulations, and generates a raw HTML report template. The user can then add their personal qualitative football analyst interpretations directly into the pipeline to compile the final tactical report.

Currently, the `report/` folder contains two finalized and very opposite examples of these network resilience outputs, complete with my personal match analyst insights on Éver Banega (Sevilla FC) and Lucas Biglia (AC Milan) impacts on their respective teams.

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

### Prerequisites
Ensure you have **Python 3.12.4** installed along with the required libraries imported across the project modules. Before executing the pipeline, make sure to download Pappalardo's event data JSON files and place them exactly as shown in the *Project Structure* section.

### Core Architecture & Pipeline (`app.py`)
The entire analysis workflow is contained within `app.py` and divided into 5 cells, designed to be executed sequentially in an **interactive window** (such as VS Code's Interactive Python environment):

* **Cell [1] & [2]:** Imports the required packages and opens an interactive widget for team selection.
* **Cell [3] & [4]:** Computes passing total weights, displays the top performers in terms of passing weights per 90 minutes, and allows the user to select the playmaker and build the starting XI lineup.
* **Cell [5]:** Runs the targeted attack simulation and generates the final HTML output report where the user can add their own interpretations.

## 📊 Metrics Computed and Design Choices Explanation

* **Pass Receiver Evaluation:** Since the Pappalardo et al. dataset does not explicitly document the recipient of a pass, a sequential proximity logic was implemented. A pass is credited as successful to a specific teammate if the immediate subsequent chronological event involves that same player.

* **Passing Weights Formula ($w$):** To balance both pass quantity and pass quality, the weight formula scales progressive threat. It is calculated as:
  $$w = \alpha \cdot \Delta_{value} + \beta \cdot End_{value}$$
  Where $\Delta_{value}$ represents the ball progression toward a more dangerous zone (calculated via a non-linear distance-to-goal function as $End_{danger} - Start_{danger}$), and $End_{value}$ measures the absolute danger level of the target zone. The parameters $\alpha = 0.6$ and $\beta = 0.4$ were chosen rationally to prevent the metric from biasing strictly towards either high-frequency safe passes or low-frequency high-risk long balls.

* **Interactive Formation Builder:** To build a realistic formation network, the script filters all matches where the chosen playmaker was in the starting XI. It extracts the team's most frequent tactical scheme (e.g., 4-3-3) and populates it by selecting the players who shared the highest number of minutes on the pitch with the playmaker. This design choice guarantees that the simulated baseline represents the most stable tactical environment of the team. In case the chosen formation presents some inconsistencies, the user is able substitute one or more players.

* **Collective Passing Output Decay:** This is the first of the metrics presented in the output file and it simply calculates the percentage of the sum of passing weights per 90 minutes attributable to the playmaker..  As a personal interpretation rule a collective decay exceeding **15-16%** could indicate that the playmaker is structurally vital to the team's transition phase.

* **Dangerous Passing Lanes:** These tables extract the top 10 strongest edge connections (sender $\rightarrow$ receiver by weight per 90 shared minutes) with and without the presence of the playmaker. Comparing these highlights which passing lanes emerge as best alternatives going forward without going through the playmaker.

* **Weighted In-Degree (Involvement Decay):** Measures the total progressive pass value a player receives from their teammates. Evaluating the percentage change before and after the playmaker's isolation exposes dependencies to him. A high negative decay indicates that a player offensive involvement is highly dependant on the playmaker's distribution. I personally could interpret a value over **30%** of a specific player as a potential indicator of a very high reliance on the playmaker.

* **Forward Receptions in the Danger Zone:** A purely empirical, observed spatial plot capturing every pass received by a forward inside a specified radius from the opponent's goal ($Danger\ Zone$). The "After" condition filters out all direct passes originating from the isolated playmaker. This option was preferred to a shortest path length type construct as with such dense graphs it is hard to compute an interpretable similar metric.

---


## 👥 Author
_Livio Guerra - Leiden University (Student ID: s4444159)_  

_Email: l.guerra@umail.leidenuniv.nl_