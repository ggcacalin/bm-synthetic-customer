from dotenv import load_dotenv
import pymssql
import pandas as pd
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader

load_dotenv()

# SQL Server credentials
sql_prod_db_user = os.environ.get("SQL_PROD_USER")
sql_prod_db_pass = os.environ.get("SQL_PROD_PASS")

csv_path = os.environ.get("FILE_ORIGIN") + os.environ.get("SQL_CSV_DF")
faiss_path = os.environ.get("FILE_ORIGIN") + os.environ.get("SQL_FAISS")

# Store mosaic names and title scores in vector database

# Connection to SQL Server
def get_connector(sql_server, sql_db):
    conn = pymssql.connect(
        server=sql_server,
        user=sql_prod_db_user,
        password=sql_prod_db_pass,
        database=sql_db,
        as_dict=True
    )
    cursor = conn.cursor()
    return cursor

# Method to get a query result as a pandas dataframe
def query(sql_query):
    cursor.execute(sql_query)
    records = cursor.fetchall()
    output_df = pd.DataFrame(records)
    return output_df

# Initialising DB browser
cursor = get_connector("bm-campaign-db.database.windows.net", "bm-campaign-db")

# Big query for title scores
mosaic_title_scores_query = """
select a.MosaicID, sum(a.Score) as Score, sum(a.IndexScore) as IndexScore, sum(a.Sample) as Sample, sum(a.Weighted000) as Weighted000, b.Name as MosaicName, b.Description, b.Keywords, d.Name as TitleType, e.Name as TitleCategory
from
(select min(ID) as ID, Name, Description, Keywords from Mosaics group by Name, Description, Keywords) as b
JOIN (select TitleID, MosaicID, max(ID) as ID, max(Score) as Score, max(IndexScore) as IndexScore, max(Sample) as Sample, max(Weighted000) as Weighted000
from TitleScores
group by MosaicID, TitleID) as a
ON a.MosaicID = b.ID
JOIN Titles as c
ON a.TitleID = c.ID
JOIN (select Name, ID, MediaCategoryID from MediaType where Name != 'Quintile') as d
ON c.MediaTypeID = d.ID
JOIN MediaCategory as e
ON d.MediaCategoryID = e.ID
group by a.MosaicID, b.Name, b.Description, b.Keywords, d.Name, e.Name
order by MosaicID asc, Score desc
"""

mosaic_title_scores_df = query(mosaic_title_scores_query)

#Sort out mosaic name
mosaic_title_scores_df['Name'] = mosaic_title_scores_df['Keywords']
mosaic_title_scores_df['Name'] = mosaic_title_scores_df['Name'].fillna(mosaic_title_scores_df['Description'])
mosaic_title_scores_df['Name'] = mosaic_title_scores_df['Name'].fillna(mosaic_title_scores_df['MosaicName'])
mosaic_title_scores_df.drop(columns = ['MosaicName', 'Description', 'Keywords', 'IndexScore', 'Sample', 'Weighted000'], inplace = True)

# Further processing & Save as csv
bad_name_list = ['test', 'null']
mosaic_title_scores_df.drop(mosaic_title_scores_df[mosaic_title_scores_df['Name'].isin(bad_name_list)].index, inplace = True)
mosaic_title_scores_df.to_csv(csv_path, index = False, encoding = 'utf-8')

# Load tuples from SQL DB as vectorstore documents
mosaic_docs = []
loader = CSVLoader(csv_path)
documents = loader.load()
mosaic_docs.extend(documents)

mosaic_db = FAISS.from_documents(mosaic_docs, OpenAIEmbeddings())
mosaic_db.save_local(faiss_path)

print(mosaic_title_scores_df.head())