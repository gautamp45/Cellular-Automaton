import pandas as pd
from bs4 import BeautifulSoup
import requests


wikiurl = 'https://conwaylife.com/wiki/List_of_Life-like_rules'
table_class = "wikitable sortable jquery-tablesorter"
response = requests.get(wikiurl)
print(response.status_code)

soup = BeautifulSoup(response.text, 'html.parser')
ruletable = soup.find('table', {'class': "wikitable"})

df = pd.read_html(str(ruletable))
df = pd.DataFrame(df[0])
df = df.drop(columns=['Unnamed: 3'])
print(df.head())
