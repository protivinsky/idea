import os
import pandas as pd
from pathlib import Path
import requests
import bs4


url_parent = "https://dsia.msmt.cz/vystupy/region/"
url_template = url_parent + "vu_region{}.html"
years = range(2013, 2024)
dfs = {}

def download_file(url, local_filename):
    response = requests.get(url)
    if 200 <= response.status_code < 300:
        Path(local_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url} to {local_filename} with status {response.status_code}")

download = True
data_root = Path("data") / "dsia"
(data_root / "meta").mkdir(parents=True, exist_ok=True)

for year in years:
    print("Processing year", year)
    url = url_template.format(year)
    response = requests.get(url)
    response.encoding = "windows-1250"
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    rows = []
    for row in table.find_all("tr")[1:]:
        cols = []
        for col in row.find_all(["td", "th"]):
            cols.append(col.text)
            if download and hasattr(col.a, "href"):
                file_url = col.a["href"]
                dir_part, file_part = file_url.split("/")
                local_filename = data_root / dir_part / str(year) / file_part
                if local_filename.exists():
                    print("File", local_filename, "already exists, skipping")
                    continue
                print("Downloading file", file_url)
                download_file(
                    url_parent + file_url,
                    data_root / dir_part / str(year) / file_part
                )
        rows.append(cols)

    df = pd.DataFrame(rows, columns=["id", "form", "instr", "data"])
    df = df.set_index("id")
    df.to_csv(data_root / "meta" / f"{year}.csv")

