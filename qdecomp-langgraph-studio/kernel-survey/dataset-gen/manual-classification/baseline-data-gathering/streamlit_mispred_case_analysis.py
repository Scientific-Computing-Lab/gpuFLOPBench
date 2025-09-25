
import streamlit as st
from sqlite_helper import sqlitefile_to_dataframe
import pandas as pd
import os

def get_sqlite_files():
    db_dir = './checkpoints/'

    # get all the sqlite files in the db_dir
    sqlite_files = [f for f in os.listdir(db_dir) if f.endswith('.sqlite')]

    for sqlite_file in sqlite_files:
        print(f"Processing [{sqlite_file}]") 

    # keep only the ones with gpt-5-mini in the name
    sqlite_files = [f for f in sqlite_files if 'gpt-5-mini' in f]
    print(sqlite_files)

    sqlite_files = [os.path.join(db_dir, f) for f in sqlite_files]
    return sqlite_files

def load_dataframes(full_paths: list[str]):

    df = pd.DataFrame()

    for sqlite_file in full_paths:
        filedf = sqlitefile_to_dataframe(sqlite_file)
        df = pd.concat([df, filedf], ignore_index=True)

    return df

def main():
    st.title("Misprediction Case Analysis")

    sqlite_files = get_sqlite_files()
    df = load_dataframes(sqlite_files)




if __name__ == "__main__":
    main()
