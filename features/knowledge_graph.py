# src/knowledge_graph.py
import spacy
import networkx as nx
from pyvis.network import Network

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
def extract_entities(text):
    """
    Extract important named entities
    """
    doc = nlp(text)

    entities = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT"}
        and len(ent.text) > 2
    ]

    return list(set(entities))  # remove duplicates


def build_graph(chunks):
    """
    Build a cleaner knowledge graph
    """

    G = nx.Graph()

    # Limit chunks to avoid massive graphs
    for chunk in chunks[:25]:

        ents = extract_entities(chunk)

        # Connect only nearby entities
        for i in range(len(ents) - 1):

            e1 = ents[i]
            e2 = ents[i + 1]

            if e1 != e2:

                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    return G


def draw_graph(G):

    net = Network(
        height="650px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
        notebook=False
    )

    # Enable interactive controls
    net.show_buttons(filter_=['physics'])

    # Add nodes
    for node in G.nodes:
        net.add_node(
            node,
            label=node,
            title=node,        # hover tooltip
            size=20
        )

    # Add edges
    for source, target, data in G.edges(data=True):
        weight = data.get("weight", 1)

        net.add_edge(
            source,
            target,
            value=weight,
            title=f"Relationship strength: {weight}"
        )

    # Physics for better layout
    net.repulsion(
        node_distance=200,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )

    # Save graph
    net.save_graph("kg.html")

    with open("kg.html", "r", encoding="utf-8") as f:
        html = f.read()

    return html