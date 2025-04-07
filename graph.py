# %% [markdown]
# carregar libraries

# %%
import matplotlib
matplotlib.use('Agg')

import networkx as nx
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import json
from itertools import islice

def rgb_string_to_hex(rgb_string):
    rgb_values = rgb_string.strip('rgb()').split(',')
    rgb = tuple(int(value.strip()) for value in rgb_values)
    return '#{:02X}{:02X}{:02X}'.format(rgb[0], rgb[1], rgb[2])

# %% [markdown]
# tratar dados e definir intervalos do sentimento

# %%
# Dependecie for: how many words and sentiment intervals

def data_insights(data_in, globalVar):

    sentiments = []
    for word in data_in:
        sentiments.append(data_in[word]["sentiment"])

    globalVar["all_sentiments"] = {"q10": np.quantile(sentiments, 0.1),
                               "q30": np.quantile(sentiments, 0.3),
                               "q70": np.quantile(sentiments, 0.7),
                               "q90": np.quantile(sentiments, 0.9)}

def data_filter(data_in, numero_de_palavras, globalVar):

    # Filter the data
    if numero_de_palavras >= len(data_in):
        data = data_in
    else:
        data = dict(islice(sorted(data_in.items(), key=lambda item: item[1]["count"], reverse=True), numero_de_palavras))

    # Dependecie for: sentiment intervals
    sentiments2 = []
    counts2 = []
    for word in data:
        sentiments2.append(data[word]["sentiment"])
        counts2.append(data[word]["count"])

    # Set sentiment intervals
    quantile10 = globalVar["all_sentiments"]["q10"]*0.4 + np.quantile(sentiments2, 0.1)*0.6
    quantile30 = globalVar["all_sentiments"]["q30"]*0.3 + np.quantile(sentiments2, 0.3)*0.7
    quantile70 = globalVar["all_sentiments"]["q70"]*0.3 + np.quantile(sentiments2, 0.7)*0.7
    quantile90 = globalVar["all_sentiments"]["q90"]*0.4 + np.quantile(sentiments2, 0.9)*0.6

    globalVar["data_filtered"] = data
    globalVar["sentiment_intervals"] = {"q10": quantile10,
                                        "q30": quantile30,
                                        "q70": quantile70,
                                        "q90": quantile90}
    
    globalVar["min_count"] = min(counts2)

# %% [markdown]
# criar a base do grafo

# %%

def initialize_graph(data, globalVar):

    # Create the graph
    G = nx.Graph()

    #np.random.seed(21)
    spread_x = 300
    spread_y = 150
    min_distance = 50

    pos = {}
    G.add_node("center")
    pos["center"] = (0, 0)

    # Add nodes to the graph with attributes and positions
    for word, attributes in data.items():
        G.add_node(word, **attributes)

        while True:
            x = np.random.uniform(-spread_x, spread_x)
            y = np.random.uniform(-spread_y, spread_y)

            distance_from_center = np.linalg.norm([x, y])

            if distance_from_center >= min_distance:
                pos[word] = (x, y)
                break

    # Node positions
    #pos = nx.spring_layout(G, seed=124348)
    pos = nx.spring_layout(G, pos=pos, fixed=["center"], k=0.1, iterations=150, scale=1000, seed=21)

    G.remove_node("center")
    del pos["center"]

    globalVar["G"] = G
    globalVar["pos"] = pos

# %% [markdown]
# informacoes para cada node

# %%
def node_info(node,
              node_x, node_y,
              node_text, node_form,
              node_size, node_color,
              node_hovertext, custom_data,
              query, globalVar):
    
    G = globalVar["G"]
        
    # Node position
    x, y = globalVar["pos"][node]
    node_x.append(x)
    node_y.append(y)

    # Node text (name)
    if " " in node:
        splitted_text = node.split(" ")
        mid_text = len(splitted_text)//2
        node_text.append(' '.join(splitted_text[:mid_text]) + '<br>' + ' '.join(splitted_text[mid_text:]))
    else:
        node_text.append(node)

    # Node form
    node_form.append("circle")

    # Node size
    node_size.append(((np.log(G.nodes[node]["count"]/globalVar["min_count"]))*3)**1.5 + 50)
    
    # Node color
    sentiment = G.nodes[node]["sentiment"]
    if sentiment <= globalVar["sentiment_intervals"]["q10"]:
        sentiment_class = "muito negativo"
        color = "rgb(204, 0, 0)"
        node_color.append(color)
    elif sentiment <= globalVar["sentiment_intervals"]["q30"]:
        sentiment_class = "negativo"
        color = "rgb(239, 83, 80)"
        node_color.append(color)
    elif sentiment < globalVar["sentiment_intervals"]["q70"]:
        sentiment_class = "neutro"
        color = "rgb(204, 204, 204)"
        node_color.append(color)
    elif sentiment < globalVar["sentiment_intervals"]["q90"]:
        sentiment_class = "positivo"
        color = "rgb(102, 187, 106)"
        node_color.append(color)
    else:
        sentiment_class = "muito positivo"
        color = "rgb(0, 200, 81)"
        node_color.append(color)

    # Dependecie for: Node hovertext (last said) and Custom data (websites and plot)
    websites = sorted(G.nodes[node]["news"], reverse=True)
    first_website = websites[-1].split("/")
    first_website_date = first_website[5][:4] + "/" + first_website[5][4:6]
    last_website = websites[0].split("/")
    last_website_date = last_website[5][:4] + "/" + last_website[5][4:6]

    # Node hovertext
    node_hovertext.append(
        f"""Tópico: {node}
        <br>Menções: {int(G.nodes[node]['count'])}
        <br>Último registo: {last_website_date}"""
    )

    # Dependecie for: Custom data (websites)
    websites_data = ""
    for website in websites:
        website_ = website.split("/")
        websites_data += f"<p><a href='{website}' target='_blank'>{website_[5][:4]+'/'+website_[5][4:6]+' - '+ '/'.join(website_[6:])}</a></p>"

    # Dependecie for: Custom data (plot)
    times_said_by_year = {str(k): 0 for k in range(int(first_website_date[:4]),
                                            int(last_website_date[:4])+1)}
    for key in G.nodes[node]["date"].keys():
        times_said_by_year[str(key)[:4]] += int(G.nodes[node]["date"][key])
    plt.figure(figsize=(6, 4))
    plt.bar(times_said_by_year.keys(), times_said_by_year.values(), color=rgb_string_to_hex(color))
    plt.xlabel('Ano')
    plt.ylabel('Número de Menções')
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    plt.close() 
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    # Dependecie for: Custom data (source)
    source = G.nodes[node]["source"]
    source_data = ""
    for key in source.keys():
        if source[key] is not None:
            source_data += f"<li>{key}: {int(source[key])}</li>"
        
    # Node custom data
    custom_data.append(f"""
        <h2 style="text-align: center;">Associação entre<br>{query} e {node}</h2>
        <p>Menções: {int(G.nodes[node]['count'])}</p>
        <p>Sentimento: {sentiment_class}</p>
        <p>Fontes:</p>
        <ul>
        {source_data}
        </ul>
        <div class="url-navigation">
            <button onclick="navigateUrl(-1)">&lt;</button>
            <span id="current-url"></span>
            <button onclick="navigateUrl(1)">&gt;</button>
        </div>
        <div id="website-urls">
            {websites_data}
        </div>
        <img src="data:image/png;base64,{img_str}" alt="Bar Plot" style="width:100%; height:auto;">
        """)
    
def populate_nodes(G, query, globalVar):

    # Lists for node info
    node_x, node_y = [0], [0]
    node_text, node_form = [f"<b>{query}</b>"], ["square"]
    node_size, node_color= [0], ["rgb(217, 238, 252)"]
    node_hovertext, custom_data = ["Explora os tópicos ao clicares neles!"], [""]

    # Populate node information
    for node in G.nodes:
        node_info(node, #has all the information
                  node_x,
                  node_y,
                  node_text,
                  node_form,
                  node_size,
                  node_color,
                  node_hovertext,
                  custom_data,
                  query,
                  globalVar)
    
    globalVar["node_x"] = node_x
    globalVar["node_y"] = node_y
    globalVar["node_text"] = node_text
    globalVar["node_form"] = node_form
    globalVar["node_size"] = node_size
    globalVar["node_color"] = node_color
    globalVar["node_hovertext"] = node_hovertext
    globalVar["custom_data"] = custom_data

# %% [markdown]
# gerar grafo e algumas definicoes

# %%
# Create the Plotly figure
def create_graph(globalVar):

    fig = go.Figure()

    # Draw nodes
    fig.add_trace(
        go.Scatter(
            x=globalVar["node_x"],
            y=globalVar["node_y"],
            mode="markers+text",
            text=globalVar["node_text"],
            hovertext=globalVar["node_hovertext"],
            marker=dict(
                color=globalVar["node_color"],
                symbol=globalVar["node_form"],
                size=globalVar["node_size"],
                line=dict(color="black", width=1)
            ),
            hoverinfo="text",
            customdata=globalVar["custom_data"],
            hoverlabel=dict(
                font=dict(color="rgb(48, 62, 92)"),
                bordercolor="rgb(48, 62, 92)"
            )
        )
    )

    # Update the layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgb(217, 238, 252)",
        paper_bgcolor="rgb(217, 238, 252)",
        #font=dict(
        #    family="system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif",
        #)
    )

    # Generate the base HTML with graph
    html_code = fig.to_html(include_plotlyjs='inline',
                            full_html=True,
                            config={
                                'displaylogo': False,
                                'modeBarButtonsToRemove': [
                                    'select2d',
                                    'lasso2d',
                                    'resetScale2d',
                                    'toImage',
                                ]
                            })

    return html_code
# %% [markdown]
# algumas alteracoes no grafo, com css e js

# %%
# Combine the HTML and CSS/JS code
def combine_graph_html(html_code, additional_html):

    final_html = html_code.replace("</body>", additional_html + "</body>")

    return final_html

    

# Some CSS and JavaScript code to create the info panel
additional_html = """
<style>
    body, html {
        height: 100vh !important;
        width: 100vw !important;
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
    }
    .plotly-graph-div{
        height:100vh !important;
        width:100vw !important;
    }
    #info-panel {
        position: absolute;
        overflow-y: auto;
        right: 0; /* Default to right */
        top: 0;
        width: 300px; /* Fixed width of the panel */
        height: 100%; /* Full height of container */
        background-color: rgb(217, 238, 252);
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        transform: scale(0); /* Hide initially */
        transition: transform 0.3s ease; /* Smooth slide in */
        z-index: 2; /* Panel above the graph */
        display: flex;
        flex-direction: column; /* Arrange children vertically */
        color: rgb(48, 62, 92);
    }
    #info-panel.open {
        transform: scale(1); /* Slide in the panel */
    }
    .close-button {
        background-color: #ff4c4c;
        color: white;
        border: none;
        padding: 10px;
        cursor: pointer;
        float: right;
        margin-bottom: -15px;
    }
    .close-button:hover {
        background-color: #e04343;
    }
    #website-urls {
        flex: 1; /* Take remaining vertical space */
        max-height: auto; /* Adjust height as needed */
        overflow-x: auto;
        padding: 4px; /* Padding for inner content */
        background-color: #ffffff; /* Background color */
        border: 1px solid #ddd; /* Border for the scrollable area */
        box-shadow: inset 0 0 5px rgba(0,0,0,0.1); /* Inner shadow */
        margin-top: 7px; /* Spacing from the title */
    }
    #website-urls p {
        white-space: nowrap; /* Prevents line breaks within the item */
    }
    p {
        margin: 5px 5px;
    }
    ul {
        margin-top: 0; /* Set the top margin of the unordered list to 0 */
    }
    li {
        margin-bottom: 5px;
    }
    .url-navigation button {
        border: none;
        border-radius: 2px;
        cursor: pointer;
        background-color: #6c757d;
        color: white;
        transition: background-color 0.3s;
    }
    .url-navigation button:hover {
        background-color: #5a6268;
    }
</style>

<div id="info-panel">
    <button class="close-button" onclick="closePanel()">Fechar</button>
    <p id="node-info">Escolha um nó para ver detalhes.</p>
</div>

<script>
    let currentUrlIndex = 0; // Global variable to track the currently displayed URL

    // Function to close the info panel
    function closePanel() {
        var panel = document.getElementById('info-panel');
        if (panel.classList.contains('open')) {
            panel.style.transform = 'scale(0)';
            setTimeout(() => {
                panel.classList.remove('open'); 
            }, 300); 
        }
    }

    function navigateUrl(direction) {
        const urls = document.querySelectorAll("#website-urls p");
        if (urls.length === 0) return;

        // Hide the currently visible URL
        urls[currentUrlIndex].style.display = "none";

        // Update the index based on the direction
        currentUrlIndex += direction;

        // Wrap around if out of bounds
        if (currentUrlIndex < 0) {
            currentUrlIndex = urls.length - 1; // Go to the last URL if moving left
        } else if (currentUrlIndex >= urls.length) {
            currentUrlIndex = 0; // Go back to the first URL if moving right
        }

        // Show the new URL
        urls[currentUrlIndex].style.display = "block";

        // Update the display span with the current index
        document.getElementById("current-url").textContent = `Notícia ${currentUrlIndex + 1}/${urls.length}`;
    }

    // Function to initialize the URL display for a new node
    function initializeUrls() {
        const urls = document.querySelectorAll("#website-urls p");
        urls.forEach((url) => {
            url.style.display = "none"; // Hide all URLs initially
        });

        // Reset index and show the first URL if available
        currentUrlIndex = 0; 
        if (urls.length > 0) {
            urls[currentUrlIndex].style.display = "block"; // Show the first URL
        }
        
        // Update the display span with the initial index
        document.getElementById("current-url").textContent = `Notícia ${currentUrlIndex + 1}/${urls.length}`;
    }

    document.addEventListener('DOMContentLoaded', function() {
        var plotDiv = document.querySelector('.plotly-graph-div');
        
        if (plotDiv) {
            plotDiv.on('plotly_click', function(data) {
                if (data.points.length > 0) {
                    var point = data.points[0];
                    var customData = point.customdata;
                    var nodeX = point.x;
                    var nodeY = point.y

                    if (nodeX === 0 && nodeY === 0) {
                        return;
                    }

                    document.getElementById('node-info').innerHTML = customData; // Use custom data for the panel

                    // Reset the URL index when a new node is clicked
                    currentUrlIndex = 0; // Reset to first URL
                    try {
                        initializeUrls(); // Reinitialize the URLs
                    } catch (error) {
                        // Code to handle the error
                        console.error("An error occurred:", error);
                    }

                    // Determine mouse click position for panel placement
                    var mouseX = data.event.clientX; 
                    var panel = document.getElementById('info-panel');

                    // Adjust panel position based on mouse click
                    if (mouseX > window.innerWidth / 2) {
                        panel.style.left = '0'; 
                        panel.style.right = 'auto'; 
                    } else {
                        panel.style.right = '0'; 
                        panel.style.left = 'auto'; 
                    }

                    panel.classList.add('open'); 
                    panel.style.transform = 'translateX(0)'; // Animate to the open position
                }
            });
        } else {
            console.error("Graph div not found.");
        }
    });
</script>
"""

# %% [markdown]
# funcao geral
# %%

def create_keyword_graph(data_in, numero_de_palavras, query):
    globalVar = {}

    # no data available
    if len(data_in) == 0:
        globalVar["node_x"], globalVar["node_y"] = [0], [0]
        globalVar["node_text"], globalVar["node_form"] = [f"<b>{query}</b>"], ["square"]
        globalVar["node_size"], globalVar["node_color"] = [0], ["rgb(217, 238, 252)"]
        globalVar["node_hovertext"], globalVar["custom_data"] = ["Não foram encontrados tópicos relevantes..."], [""]
        html_code = create_graph(globalVar)
        final_html = combine_graph_html(html_code, additional_html)
        return final_html
    
    # data available
    if query in data_in:
        del data_in[query]
    data_insights(data_in, globalVar)
    data_filter(data_in, numero_de_palavras, globalVar)
    initialize_graph(globalVar["data_filtered"], globalVar)
    populate_nodes(globalVar["G"], query, globalVar)
    html_code = create_graph(globalVar)
    final_html = combine_graph_html(html_code, additional_html)

    #with open("graph_galptest.html", 'w') as f:
    #    f.write(final_html)

    return final_html

#%%

if __name__ == "__main__":

    with open("cache/7ab9dd2c21.json", "r") as f:
        data_in = json.load(f)

    numero_de_palavras = 200

    query = "QUERY"

    create_keyword_graph(data_in, numero_de_palavras, query)