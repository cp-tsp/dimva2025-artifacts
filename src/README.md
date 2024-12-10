
## License 

Copyright (c) 2024 [**Redacted**]  
This code is released under the terms of the MIT license. See the `LICENSE.txt` file for more information.


## Installation 
- Create a venv : `python3 -m venv .venv; source .venv/bin/activate`
- Install requirements and tools : `pip install -e .`

## Scripts

#### combine_csv.py
Combine and labelize input CSV files. If the file name starts with `cs`,
it will be labeled as "Cobalt Strike" (malicious).
Otherwise, it will be labeled as "Benign".

#### prepare_dataframe.py
Take a CSV file as input and returns only the features selected by the `-fg` option.  
The feature groups are defined in the `features.json` file.

#### learning_curves_build.py
Build a learning curve from a labeled CSV. Returns the values and statistics for the different number of training in
a JSON file.

#### learning_curves_plot_article.py
Plot the learning curves from several JSON files. Used with the JSON returned by `learning_curves_build`.

#### boxplot_article.py
Plot the boxplots from several JSON files. Used with the JSON returned by `learning_curves_build`.

#### feature_importance_build.py
Build the importance of the different features after splitting a CSV, train a model and testing it.

#### feature_importance_plot_article.py
Plot the learning curves from several JSON files. Used with the JSON returned by `build_feature_importance`.

#### plot_legends.py
Plot the legends used in the paper. The colors, labels and markers are defined in `config.json`.


## How to reproduce the results

1) Combine the CSV files using `combine_csv.py`. You can use the `-i` option to list all the different files or the `-d` option to use all the files inside a directory.   
Ex: `python3 combine_csv.py -d folder_of_csv/ -o output/path/file.csv`

2) Take the output of the previous command and prepare it with the desired features such as "netflowv5b_no_duration". Feature groups available are in `features.json`  
Ex: `python3 prepare_dataframe.py -i output/path/file.csv -fg netflowv5b_no_duration -o output/path/file_prepared.csv`

3) Compute the learning curve with the prepared CSV with `learning_curves_build.py`.  Use the `-j` option for multiprocessing.    
Ex: `python3 learning_curves_build.py -i output/path/file_prepared.csv -c ml_config_detailed.toml -l rf --seed 0 --score f1 -j 4 -o output/path/` 

4) Use `learning_curves_plot_article.py` and `boxplot_article.py` on the computed JSON files to visualize the results.   
Ex:
    - `python3 learning_curves_plot_article.py -i output/path/file_prepared_rf_f1_lc.json -o output/path/learning_curve.pdf` 
    - `python3 boxplot_article.py -i output/path/file_prepared_rf_f1_lc.json -o output/path/boxplot.pdf` 

5) Compute the feature importances with the prepared CSV with `feature_importance_build.py`.  Use the `-j` option for multiprocessing.     
Ex: 
`python3 feature_importance_build.py -c ml_config_detailed.toml -i output/path/file_prepared.csv -l rf --seed 0 --score f1 -j 4 -o output/directory/path/` 

6) Use `feature_importance_plot_article.py` on the computed JSON file to visualize the results.   
Ex: `python3 boxplot_article.py -i output/path/file_prepared_rf_f1_fi.json -o output/path/feature_importance.pdf`
