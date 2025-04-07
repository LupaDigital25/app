# Flask
from flask import Flask, render_template, request
from flask_socketio import SocketIO

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

# Others
import time
import os
import hashlib
import json
import random

# Local
from graph import create_keyword_graph
from info import pie_newsSources, timeseries_news, topic_wordcloud
from info2 import ts_topicrelation, sources_topicrelation, news_topicrelation

# testing
#from flask import redirect, url_for



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("News App") \
    .config("spark.ui.enabled", "false") \
    .getOrCreate()

# Define the data schema
schema = StructType([
    StructField("timestamp", IntegerType(), True),
    StructField("source", StringType(), True),
    StructField("archive", StringType(), True),
    StructField("id", IntegerType(), True),
    StructField("probability", FloatType(), True),
    StructField("keywords", MapType(StringType(), IntegerType()), True),
    StructField("sentiment", FloatType(), True)
])
df = spark.read.format("json").schema(schema).load("../data/news/status=success")

globalVar = {
            "search_done": False,
            "zero_results": True,
            "topicrelation": False,
            "total_amount_of_news": df.count(), # substituir por contagem "manual"
            "first_news": 1998,
            "last_news": str(df.agg(F.max("timestamp")).collect()[0][0]), # substituir por ano à mão
            }

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html', globalVar=globalVar)

@app.route('/sobre')
def sobre():
    global globalVar
    if globalVar["search_done"] == False:
        return render_template('404.html', globalVar=globalVar)
    
    return render_template('info.html', globalVar=globalVar)

@app.route('/grafo')
def grafo():
    global globalVar
    if globalVar["search_done"] == False or globalVar["zero_results"] == True:
        return render_template('404.html', globalVar=globalVar)

    return render_template('graph.html', globalVar=globalVar)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

@app.route('/pesquisa', methods=['GET'])
def pesquisa():
    global globalVar
    globalVar["search_done"] = True
    globalVar["zero_results"] = False
    globalVar["topicrelation"] = False

    # query requested
    query = request.args.get('topico', '')
    globalVar['query'] = query

    # data filtering
    query_col_counts = F.col("keywords").getItem(query)
    df_with_q = df.filter(query_col_counts.isNotNull() & (query_col_counts > 4)).cache()
    globalVar['query_amountofnews'] = df_with_q.count()

    # more then 0 news with the query?
    if globalVar['query_amountofnews'] == 0:
        globalVar['keywords'] = {}
        globalVar["zero_results"] = True
        globalVar["wordcloud"] = topic_wordcloud({},
                                                query, "static/Roboto-Black.ttf")

        # render the index page
        return render_template('info.html', globalVar=globalVar)
    
    # data exploration
    query_firstnews = str(df_with_q.agg(F.min("timestamp")).collect()[0][0])
    meses = {
        "01": "janeiro", "02": "fevereiro", "03": "março", "04": "abril",
        "05": "maio", "06": "junho", "07": "julho", "08": "agosto",
        "09": "setembro", "10": "outubro", "11": "novembro", "12": "dezembro"}
    globalVar["query_firstnews"] = f"{meses[query_firstnews[4:]]} de {query_firstnews[:4]}"

    # query already processed?
    hashed_query = hashlib.sha256(query.encode()).hexdigest()[:10]
    if os.path.exists(f"cache/{hashed_query}.json"):
        
        with open(f"cache/{hashed_query}.json", 'r') as json_file:
            globalVar['keywords'] = json.load(json_file)
    
    else:
        # process the news if not processed yet
        result = (
            df_with_q.rdd
            .flatMap(lambda row: [
                (key, (value,
                    {row["timestamp"]: value},
                    row["sentiment"]*value,
                    {row["source"]: 1},
                    [row["archive"]])) for key, value in row["keywords"].items()
            ])
            .reduceByKey(lambda a, b: (
                a[0] + b[0],  # Sum count values
                {ts: a[1].get(ts, 0) + b[1].get(ts, 0) for ts in set(a[1]) | set(b[1])},  # Merge timestamp counts
                a[2] + b[2],  # Sum sentiment values
                {source: a[3].get(source, 0) + b[3].get(source, 0) for source in set(a[3]) | set(b[3])},  # Merge source counts
                a[4] + b[4]  # Merge archive lists
            ))
            .collect()
        )
        # change data schema
        globalVar['keywords'] = {key: {"count": value[0],
                        "date": value[1],
                        "sentiment": value[2]/value[0],
                        "source": value[3],
                        "news": value[4]} for key, value in result}
        globalVar['keywords'] = {k: v for k, v in globalVar['keywords'].items() if v["count"] >= 5}

        # save in cache
        with open(f"cache/{hashed_query}.json", 'w') as json_file:
            json.dump(globalVar['keywords'], json_file)
    

    # create graph src code
    globalVar["graph_html"] = create_keyword_graph(globalVar['keywords'], 125, query)

    # create pie plot from news sources
    globalVar["pie_sources"] = pie_newsSources(df_with_q) 

    # create ts plot from news
    globalVar["ts_news"], globalVar["news_by_month"] = timeseries_news(df_with_q, query)

    # create wordcloud
    globalVar["wordcloud"] = topic_wordcloud({k: v["count"] for k,v in globalVar['keywords'].items()},
                                                query, "static/Roboto-Black.ttf")
    
    # disable topic relation
    globalVar["topicrelation"] = False

    # render the graph page
    return render_template('info.html', globalVar=globalVar)


@app.route('/relacao', methods=['GET'])
def relacao():
    global globalVar
    if globalVar["search_done"] == False or globalVar["zero_results"] == True:
        return render_template('404.html', globalVar=globalVar)
    globalVar["topicrelation"] = True

    # topic relation requested
    related_topic = request.args.get('entre', '')
    globalVar['related_topic'] = related_topic
    
    globalVar["topicrelation_exists"] = related_topic in globalVar['keywords']


    # return results
    if globalVar["topicrelation_exists"]:
        globalVar["count_topicrelation"] = globalVar['keywords'][related_topic]["count"]
        globalVar["sentiment_topicrelation"] = globalVar['keywords'][related_topic]["sentiment"]
        globalVar["ts_topicrelation"] = ts_topicrelation(globalVar["news_by_month"], globalVar['keywords'], related_topic, globalVar['query'])
        globalVar["sources_topicrelation"] = sources_topicrelation(globalVar['keywords'], related_topic)
        globalVar["news_topicrelation"] = news_topicrelation(globalVar['keywords'], related_topic)
    
    else:
        recomendations_amount = min(5, len(globalVar['keywords']))
        list_of_recomendations = random.sample(list(globalVar['keywords'].keys()), recomendations_amount)
        recomendation_output = ""
        for possible_topic in list_of_recomendations:
            recomendation_output += f"<a href='/relacao?entre={possible_topic}'>{possible_topic}</a>, "
        globalVar["recomendations_topicrelation"] = recomendation_output[:-2]

    
    return render_template('info.html', globalVar=globalVar, scroll_to_relation=True)
    
    

if __name__ == '__main__':
    socketio.run(app, debug=True)
