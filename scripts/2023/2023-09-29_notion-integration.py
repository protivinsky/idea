

# https://www.notion.so/my-integrations
NOTION_TOKEN = 'secret_Xkaz7DSmlXaawSvsZ6xYigyChy5sNa5xsg9MFwtdabi'
DB_TAGS_ID = '57df4b8484894f098e22ddc0aea09097'

import requests

headers = {
    'Authorization': 'Bearer ' + NOTION_TOKEN,
    'Content-Type': 'application/json',
    'Notion-Version': '2022-06-28',
}

def get_pages():
    url = f'https://api.notion.com/v1/databases/{DB_TAGS_ID}/query'
    payload = {'page_size': 100}
    response = requests.post(url, json=payload, headers=headers)
    data = response.json()
    results = data['results']
    return results


pages = get_pages()

pages[0]['properties']['Tag']['title'][0]['text']['content']

for i in range(10):
    tag = pages[i]['properties']['Tag']['title'][0]['text']['content']
    print(tag)

tags = [p['properties']['Tag']['title'][0]['text']['content'] for p in pages]
sorted(tags)




