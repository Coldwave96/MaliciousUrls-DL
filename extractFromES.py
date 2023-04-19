import os.path

from elasticsearch import Elasticsearch
from urllib.parse import urlparse

dataFromES = "Data/malicious_es.txt"


def is_hostname_url(url):
    parsed_url = urlparse(url)
    if not parsed_url.hostname or parsed_url.hostname.isdigit() or parsed_url.port is not None:
        return False
    else:
        return True


def data2file():
    es = Elasticsearch([{'host': '10.0.0.157', 'port': 9200, 'scheme': 'http'}])
    query = {
        "match": {
            "category": "恶意"
        }
    }
    maliciousUrls = es.search(index="cti-details-url", query=query, scroll='2m')

    # 处理第一次查询结果
    if os.path.exists(dataFromES):
        os.remove(dataFromES)

    scroll_id = maliciousUrls["_scroll_id"]
    for url in maliciousUrls['hits']['hits']:
        if is_hostname_url(url['_source']['name']):
            with open(dataFromES, 'a') as f:
                f.write(url['_source']['name'] + '\n')
        else:
            continue
        # print(url['_source']['name'])

    while True:
        maliciousUrls = es.scroll(scroll_id=scroll_id, scroll="2m")
        if len(maliciousUrls["hits"]["hits"]) == 0:
            # 所有结果已经遍历完成
            break

        # 处理查询结果
        for url in maliciousUrls["hits"]["hits"]:
            if is_hostname_url(url['_source']['name']):
                with open(dataFromES, 'a') as f:
                    f.write(url['_source']['name'] + '\n')
            else:
                continue
            # print(hit["_source"]['name'])

        # 获取新的scroll_id
        scroll_id = maliciousUrls["_scroll_id"]


if __name__ == '__main__':
    print("Running...")
    data2file()
    print("Done!")
