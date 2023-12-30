# Exploratory data analysis: Customer loans
## Max Thomas
Here, an exploratory data analysis is performed for a large portfolio of loan data. The analysis can be recovered by running:
```
cd code
source run
``` 
which will produce ```/results```. The main results of the analysis are described in [define_dataframe.ipynb](/results/define_dataframe.ipynb) and [investigate_loans.ipynb](/results/investigate_loans.ipynb).

Within ```/code```:
- *run* sets up the directories and environment, runs all the code, and tidies the results.
- *db_utils.py* contains python classes used to perform the analysis.
- *save_raw_data.py* gets loan data from AWS and saves it to a csv.
- *define_dataframe.py* cleans the loan data.
- *investigate_loans.py* performs the exploratory analysis.

The directory structure looks like this:
```
.
├── README.md
├── code
│   ├── db_utils.py
│   ├── define_dataframe.py
│   ├── investigate_loans.py
│   ├── run
│   └── save_raw_data.py
├── data
│   ├── loan_payments-clean.csv
│   ├── loan_payments-clean.pkl
│   └── loan_payments-raw.csv
├── results
│   ├── define_dataframe.html
│   ├── define_dataframe.ipynb
│   ├── investigate_loans.html
│   └── investigate_loans.ipynb
└── setup
    ├── credentials.yaml
    ├── environment.yaml
    └── missingno-modified.py
```

## Set up notes
Create the environment with 
```
cd setup
conda env create -f environment.yaml
```

There is a package conflict with *missingno* and *matplotlib* that needs to be fixed before running.
```
mv missingno-modified.py ~/anaconda3/envs/aicore_eda/lib/python3.12/site-packages/missingno/missingno.py
```

Solution here:
https://stackoverflow.com/questions/35970686/ansible-ssh-error-unix-listener-too-long-for-unix-domain-socket/35971053#35971053