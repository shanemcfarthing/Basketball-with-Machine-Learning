import csv, os, numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def read_csv(filename):
    "Reads thee csv file into a list of rows, with each row being a dictionary mapping column names to values."
    column_names = None
    row_data = []
    with open(filename, newline='') as in_file:
        reader = csv.reader(in_file)
        for row in reader:
            # record column names to use as labels
            if column_names is None:
                column_names = row
            # record column values
            else:
                row_values = {}
                for entry, label in zip(row, column_names):
                    row_values[label] = entry
                row_data.append(row_values)
    return row_data

def query_rows(test_function, database):
    "Returns all rows for which test_function returns true."
    results = []
    for row in database:
        if test_function(row):
            results.append(row)
    return results


# read all files in database directory other than the readme
parsed_files = {}
for filename in os.listdir("./database/"):
    if filename != "readme.txt":
        parsed_files[filename] = read_csv(f"./database/{filename}")

# # sample query, prints number of players 5 feet tall
# for player_info in parsed_files["players.txt"]:
#     if player_info["h_feet"] == "5":
#         print(player_info)

# # sample query 2, same as above sample but uses query_rows function
# print(len(query_rows(lambda player: player["h_feet"] == "5", parsed_files["players.txt"])))

# year range with shared data: 1950 - 2003, missing allstars data for 1998
# years used: 1983 - 2003
# numeric clustering features: gp,minutes,pts,dreb,oreb,reb,asts,stl,blk,turnover,pf,fga,fgm,fta,ftm,tpa,tpm

# restrictions on the years from which to take data
clustering_year_lower_bound = 1983
clustering_year_upper_bound = 2003
def clustering_year_match_p(row):
    "Returns True iff a database row matches the year restrictions for the clustering algorithm."
    year = int(row["year"])
    return clustering_year_lower_bound <= year <= clustering_year_upper_bound

# the labels of numeric features to extract for clustering
clustering_feature_labels = ["gp","minutes","pts","dreb","oreb","reb","asts","stl","blk","turnover","pf","fga","fgm","fta","ftm","tpa","tpm"]

def extract_clustering_features(dataset):
    # restrict data to desired year range
    dataset = query_rows(clustering_year_match_p, dataset)
    # extract the features from the dataset
    feature_vectors = []
    player_labels = []
    for row in dataset:
        player_id = row["ilkid"]
        feature_vector = []
        for label in clustering_feature_labels:
            feature_string = row[label]
            if feature_string == '': # if feature is missing
                feature_value = np.nan # use a missing feature marker (numpy's 'not a number' value)
            else:
                feature_value = int(feature_string) # otherwise convert the feature to a numerical value
            feature_vector.append(feature_value)
        feature_vectors.append(feature_vector)
        player_labels.append(player_id)
    # impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean").fit(feature_vectors)
    feature_vectors = imputer.transform(feature_vectors)
    ## scale the features before returning for clustering
    # first scale to mean 0 with unit variance
    scaler1 = StandardScaler().fit(feature_vectors)
    feature_vectors = scaler1.transform(feature_vectors)
    # second scale to 0,1 range for human interpretability
    scaler2 = MinMaxScaler((0, 1)).fit(feature_vectors)
    feature_vectors = scaler2.transform(feature_vectors)
    return feature_vectors, player_labels
