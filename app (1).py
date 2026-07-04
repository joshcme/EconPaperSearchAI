import os
import json
import requests
import pandas as pd
import networkx as nx
from openai import OpenAI
from networkx import bipartite
from networkx.algorithms import community as nx_community
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

# Page configuration
st.set_page_config(page_title="AI-Powered Research Explorer", layout="wide")
# Initialize all session state variables
if "step" not in st.session_state:
    st.session_state.step = "input"
if "results" not in st.session_state:
    st.session_state.results = None
if "G" not in st.session_state:
    st.session_state.G = None
if "projections" not in st.session_state:
    st.session_state.projections = None
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None
if "openalex_data" not in st.session_state:
    st.session_state.openalex_data = None
if "concept_list" not in st.session_state:
    st.session_state.concept_list = None
if "centralities" not in st.session_state:
    st.session_state.centralities = None
if "communities" not in st.session_state:
    st.session_state.communities = None
if "path_data" not in st.session_state:
    st.session_state.path_data = None
if "ai_response" not in st.session_state:
    st.session_state.ai_response = None
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "graph_figures" not in st.session_state:
    st.session_state.graph_figures = None

# Title
st.title('AI-Powered Research Explorer')
st.caption('This is where you will display the output of your **graph** and the **AI interpretation**.')

# Main input
inq = st.text_area("Type your research inquiry", height=100, key="inq_input")

# Sidebar
with st.sidebar:
    st.header("Research Parameters")
    
    sample_economics_fields = {
        "Macroeconomic Policy": ["Taxation", "Public Spending", "Interest Rate Decisions"],
        "Economic Growth": ["Firm Productivity", "Labor Supply", "Capital Investment"],
        "Inflation": ["Consumer Pricing", "Wage Setting", "Production Costs"],
        "International Economics": ["Exchange Rates", "Trade Decisions", "Foreign Investment"],
        "Labor Markets": ["Hiring Decisions", "Wage Negotiation", "Worker Training"],
        "Game Theory": ["Mechanism Design", "Auction Theory", "Multi-disciplinary Applications"]
    }
    
    macro = st.selectbox('Macro-field', sample_economics_fields.keys(), index=None, key="macro_input")
    micro = st.selectbox('Micro-field', sample_economics_fields.get(macro, []), index=None, key="micro_input")
    
    research_idea = st.text_area('Research idea', height=100, key="research_idea_input")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start date', value=datetime(2015, 1, 1), min_value=datetime(1950, 1, 1), key="start_date")
    with col2:
        end_date = st.date_input('End date', value=datetime.now(), key="end_date")
    
    # Validate dates
    if start_date and end_date and start_date > end_date:
        st.error("⚠️ Start date must be before end date.")
        submit_disabled = True
    else:
        submit_disabled = False
    
    # Graph visualization options
    st.subheader("Graph Visualization Settings")
    show_labels = st.checkbox("Show node labels", value=True, key="show_labels")
    graph_layout = st.selectbox(
        "Graph Layout",
        ["spring", "circular", "kamada_kawai", "random"],
        index=0,
        key="graph_layout"
    )
    
    submit = st.button('Run analysis', disabled=submit_disabled, type="primary", key="submit_button")

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def reconstruct_abstract(abstract_inverted_index):
    """Reconstruct abstract from inverted index."""
    if not abstract_inverted_index:
        return ""
    
    word_positions = []
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            word_positions.append((position, word))
    
    word_positions = sorted(word_positions)
    return " ".join(word for _, word in word_positions)

def search_openalex_works(query, start_date, end_date, per_page=50, api_key=st.secrets.get("OPENALEX_API_KEY")):
    """Search OpenAlex for works matching the query."""
    BASE_URL = "https://api.openalex.org"
    endpoint = f"{BASE_URL}/works"
    
    # Convert query to string if it's a list
    if isinstance(query, list):
        query = " ".join(query)
    
    params = {
        "search": query,
        "per-page": per_page,
        "filter": f"from_publication_date:{start_date},to_publication_date:{end_date}",
        "select": "id,display_name,publication_year,cited_by_count,doi,authorships,topics,primary_location,abstract_inverted_index"
    }
    
    if api_key:
        params["api_key"] = api_key
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json().get('results', [])
        
        # Reconstruct abstracts
        for paper in data:
            paper['abstract'] = reconstruct_abstract(paper.get('abstract_inverted_index', {}))
            if 'abstract_inverted_index' in paper:
                paper.pop('abstract_inverted_index')
        
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from OpenAlex: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"Error parsing OpenAlex response: {str(e)}")
        return []

def set_graph(openalex):
    """Build a NetworkX graph from OpenAlex data."""
    G = nx.Graph()
    
    if not openalex:
        return G
    
    for paper in openalex:
        # Paper nodes
        G.add_node(
            paper.get('id'),
            node_type='paper',
            label=paper.get('display_name', 'Unknown Paper'),
            year=paper.get('publication_year'),
            citation_count=paper.get('cited_by_count', 0),
            doi=paper.get('doi'),
            journal=paper.get('primary_location', {}).get('raw_source_name', 'Unknown'),
            abstract=paper.get('abstract', '')
        )
        
        # Authors and edges
        for author_data in paper.get('authorships', []):
            author_info = author_data.get('author', {})
            author_id = author_info.get('id')
            if not author_id:
                author_id = author_info.get('display_name', 'Unknown Author')
            
            G.add_node(
                author_id,
                node_type='author',
                label=author_info.get('display_name', 'Unknown Author')
            )
            
            G.add_edge(
                author_id,
                paper.get('id'),
                relationship='written_by'
            )
        
        # Concepts and edges
        for topic in paper.get('topics', []):
            topic_id = topic.get('id')
            if topic_id:
                G.add_node(
                    topic_id,
                    node_type='concept',
                    label=topic.get('display_name', 'Unknown Concept')
                )
                G.add_edge(
                    topic_id,
                    paper.get('id'),
                    relationship='discusses'
                )
    
    return G

def graph_projection(G, main_node_type, other_node_type):
    """Create a weighted projection between two node types."""
    main_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == main_node_type]
    other_nodes = [node for node, data in G.nodes(data=True) if data.get('node_type') == other_node_type]
    
    if not main_nodes or not other_nodes:
        return nx.Graph()
    
    main_other_graph = G.subgraph(main_nodes + other_nodes).copy()
    return bipartite.weighted_projected_graph(main_other_graph, main_nodes)

def compute_centrality(projG):
    """Compute centrality scores for nodes in a projected graph."""
    if not projG.nodes():
        return []
    
    connection_scores = dict(projG.degree(weight='weight'))
    connection_rows = []
    
    for node, score in connection_scores.items():
        data = {
            'id': node,
            'label': projG.nodes[node].get('label', str(node)),
            'connection_score': score
        }
        
        if projG.nodes[node].get('node_type') == 'paper':
            data['abstract'] = projG.nodes[node].get('abstract', '')
            data['citation_count'] = projG.nodes[node].get('citation_count', 0)
            data['year'] = projG.nodes[node].get('year')
        
        connection_rows.append(data)
    
    # Sort by connection score descending
    connection_rows.sort(key=lambda x: x['connection_score'], reverse=True)
    return connection_rows

def shortest_path(graph, node1, node2):
    """Find shortest path between two nodes in a graph."""
    if not graph.has_node(node1) or not graph.has_node(node2):
        return []
    
    try:
        shortest_path_nodes = nx.shortest_path(
            graph,
            source=node1,
            target=node2
        )
    except nx.NetworkXNoPath:
        return []
    
    path_rows = []
    for step, node in enumerate(shortest_path_nodes, start=1):
        data = {
            'step': step,
            'id': node,
            'label': graph.nodes[node].get('label', str(node))
        }
        
        if graph.nodes[node].get('node_type') == 'paper':
            data['abstract'] = graph.nodes[node].get('abstract', '')
            data['citation_count'] = graph.nodes[node].get('citation_count', 0)
            data['year'] = graph.nodes[node].get('year')
        
        path_rows.append(data)
    
    return path_rows

def community_detection(graph):
    """Detect communities in a graph using Louvain algorithm."""
    if not graph.nodes():
        return []
    
    try:
        communities = nx_community.louvain_communities(
            graph,
            weight="weight",
            seed=42
        )
    except Exception as e:
        st.warning(f"Community detection failed: {str(e)}")
        return []
    
    # Create a copy to avoid modifying the original graph
    graph_copy = graph.copy()
    for community_id, community in enumerate(communities, start=1):
        for node in community:
            graph_copy.nodes[node]["community"] = community_id
    
    community_lookup_rows = []
    for node in graph_copy.nodes:
        data = {
            'id': node,
            'label': graph_copy.nodes[node].get('label', str(node)),
            "community": graph_copy.nodes[node].get("community", 0)
        }
        
        if graph_copy.nodes[node].get('node_type') == 'paper':
            data['abstract'] = graph_copy.nodes[node].get('abstract', '')
            data['citation_count'] = graph_copy.nodes[node].get('citation_count', 0)
            data['year'] = graph_copy.nodes[node].get('year')
        
        community_lookup_rows.append(data)
    
    return community_lookup_rows

def ask_deepseek(system_prompt, user_prompt, temperature=0.3, return_json=False):
    """Query the DeepSeek API."""
    try:
        # Get API key from secrets
        api_key = st.secrets.get("DEEPSEEK_API_KEY")
        if not api_key:
            st.error("DeepSeek API key not found. Please set DEEPSEEK_API_KEY in secrets.")
            return None
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        MODEL = "deepseek-chat"  # Correct model name
        
        response_params = {
            'model': MODEL,
            'temperature': temperature,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        }
        
        if return_json:
            response_params['response_format'] = {'type': 'json_object'}
        
        response = client.chat.completions.create(**response_params)
        return response.choices[0].message.content
    
    except Exception as e:
        st.error(f"Error calling DeepSeek API: {str(e)}")
        return None

def create_network_visualization(graph, title="Network Graph", layout="spring", show_labels=True, highlight_nodes=None, highlight_edges=None):
    """Create a matplotlib visualization of the network graph."""
    if not graph or len(graph.nodes()) == 0:
        return None
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.random_layout(graph, seed=42)
    
    # Define node colors by type
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('node_type', 'unknown')
        if node_type == 'paper':
            node_colors.append('#FF6B6B')  # Red
            node_sizes.append(300)
        elif node_type == 'concept':
            node_colors.append('#4ECDC4')  # Teal
            node_sizes.append(200)
        elif node_type == 'author':
            node_colors.append('#45B7D1')  # Blue
            node_sizes.append(150)
        else:
            node_colors.append('#95A5A6')  # Gray
            node_sizes.append(100)
    
    # Highlight specific nodes if provided
    if highlight_nodes:
        for i, node in enumerate(graph.nodes()):
            if node in highlight_nodes:
                node_colors[i] = '#FFD700'  # Gold
                node_sizes[i] = 500
    
    # Draw the graph
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    edge_colors = []
    edge_widths = []
    for edge in graph.edges():
        if highlight_edges and edge in highlight_edges:
            edge_colors.append('#FFD700')  # Gold for highlighted edges
            edge_widths.append(3.0)
        else:
            edge_colors.append('#95A5A6')  # Gray
            edge_widths.append(0.5)
    
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.5)
    
    # Add labels if requested
    if show_labels:
        labels = {}
        for node in graph.nodes():
            label = graph.nodes[node].get('label', str(node))
            # Truncate long labels
            if len(label) > 30:
                label = label[:27] + "..."
            labels[node] = label
        
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=8, font_weight='bold')
    
    # Set title and remove axes
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Papers'),
        Patch(facecolor='#4ECDC4', label='Concepts'),
        Patch(facecolor='#45B7D1', label='Authors'),
    ]
    if highlight_nodes:
        legend_elements.append(Patch(facecolor='#FFD700', label='Highlighted Nodes'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_community_visualization(graph, layout="spring", show_labels=True):
    """Create a visualization with communities colored differently."""
    if not graph or len(graph.nodes()) == 0:
        return None
    
    # Detect communities if not already detected
    if not any('community' in data for _, data in graph.nodes(data=True)):
        try:
            communities = nx_community.louvain_communities(graph, weight="weight", seed=42)
            for community_id, community in enumerate(communities, start=1):
                for node in community:
                    graph.nodes[node]["community"] = community_id
        except:
            # If community detection fails, assign all nodes to same community
            for node in graph.nodes():
                graph.nodes[node]["community"] = 1
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.random_layout(graph, seed=42)
    
    # Get unique communities
    communities = set()
    for node in graph.nodes():
        communities.add(graph.nodes[node].get('community', 1))
    
    # Assign colors to communities
    import colorsys
    community_colors = {}
    for i, community in enumerate(sorted(communities)):
        hue = i / len(communities)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        community_colors[community] = rgb
    
    # Draw nodes by community
    for community in sorted(communities):
        community_nodes = [node for node in graph.nodes() if graph.nodes[node].get('community', 0) == community]
        node_color = community_colors[community]
        nx.draw_networkx_nodes(
            graph, pos, 
            ax=ax,
            nodelist=community_nodes,
            node_color=[node_color] * len(community_nodes),
            node_size=200,
            alpha=0.8
        )
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#95A5A6', width=0.5, alpha=0.3)
    
    # Add labels if requested
    if show_labels:
        labels = {}
        for node in graph.nodes():
            label = graph.nodes[node].get('label', str(node))
            if len(label) > 20:
                label = label[:17] + "..."
            labels[node] = label
        
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=7, font_weight='bold')
    
    # Set title and remove axes
    ax.set_title(f"Community Structure - {len(communities)} Communities Found", fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_path_visualization(graph, path, layout="spring", show_labels=True):
    """Create a visualization highlighting a specific path."""
    if not graph or len(graph.nodes()) == 0 or not path:
        return None
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.random_layout(graph, seed=42)
    
    # Draw all nodes
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='#95A5A6', node_size=100, alpha=0.3)
    
    # Highlight path nodes
    path_nodes = [node['id'] for node in path]
    nx.draw_networkx_nodes(
        graph, pos, 
        ax=ax,
        nodelist=path_nodes,
        node_color='#FFD700',
        node_size=300,
        alpha=0.9
    )
    
    # Draw all edges
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='#95A5A6', width=0.5, alpha=0.3)
    
    # Highlight path edges
    path_edges = []
    for i in range(len(path_nodes) - 1):
        if graph.has_edge(path_nodes[i], path_nodes[i+1]):
            path_edges.append((path_nodes[i], path_nodes[i+1]))
    
    if path_edges:
        nx.draw_networkx_edges(
            graph, pos, 
            ax=ax,
            edgelist=path_edges,
            edge_color='#FFD700',
            width=3.0,
            alpha=0.9
        )
    
    # Add labels if requested
    if show_labels:
        labels = {}
        for node in path_nodes:
            node_label = graph.nodes[node].get('label', str(node))
            if len(node_label) > 25:
                node_label = node_label[:22] + "..."
            labels[node] = node_label
        
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=8, font_weight='bold')
    
    # Set title and remove axes
    ax.set_title(f"Shortest Path - {len(path)} steps", fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add step labels
    for i, step in enumerate(path):
        node = step['id']
        x, y = pos[node]
        step_label = f"Step {i+1}"
        ax.annotate(step_label, (x, y-0.08), xytext=(0, -10), 
                   textcoords='offset points', ha='center', va='top',
                   fontsize=8, fontweight='bold', color='#2C3E50',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# Main Processing Logic
# -------------------------------------------------------------------

if submit:
    with st.spinner("Analyzing your research inquiry..."):
        # 1. Get AI analysis of the research inquiry
        system_prompt = """
        You are an AI research assistant helping economics students explore research ideas.
        
        Your task is to:
        1. Extract useful research concepts from a student's research idea.
        2. Identify which one of three functions best helps solve the user's request.
        
        The three functions are:
        - 'idea_evaluation_centrality': Show which individual papers, authors, and concepts are central to the field
        - 'concept_shortest_path': Identify how 2 papers or 2 concepts connect with each other
        - 'paper_community_detection': Identify which clusters of concepts or papers are central to the field
        
        Return valid JSON only. Return Task 1 as a list and Task 2 as a single string exactly as the label given to the functions.
        
        Use this JSON format:
        {"related_concepts": ["...", "..."], "choice": "..."}
        
        Rules:
        - Keep the "related_concepts" concepts short and searchable.
        - Include both direct keywords and related ideas.
        - Focus on economics, game theory, auctions, market design, and online marketplaces when relevant.
        - Do not invent paper titles.
        - Do not include long explanations.
        """
        
        user_prompt = f"""
        Macro-field: {macro}
        Micro-field: {micro}
        Research idea: {research_idea}
        Inquiry: {inq}
        
        Extract structured research concepts in JSON.
        """
        
        ai_response = ask_deepseek(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            return_json=True
        )
        
        if not ai_response:
            st.stop()
        
        try:
            user_choice = json.loads(ai_response)
        except json.JSONDecodeError:
            st.error("AI did not return valid JSON. Please try again with a more specific inquiry.")
            st.stop()
        
        # Validate choice
        valid_choices = ['idea_evaluation_centrality', 'concept_shortest_path', 'paper_community_detection']
        if user_choice.get('choice') not in valid_choices:
            st.error(f"Invalid AI choice: {user_choice.get('choice')}. Please try again.")
            st.stop()
        
        # Store in session state
        st.session_state.user_choice = user_choice
        
        # 2. Search OpenAlex
        with st.spinner(f"Searching OpenAlex for: {', '.join(user_choice.get('related_concepts', []))}..."):
            openalex_data = search_openalex_works(
                query=user_choice.get('related_concepts', []),
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                per_page=50
            )
            
            if not openalex_data:
                st.warning("No papers found for the given query and date range. Please try different parameters.")
                st.stop()
            
            st.session_state.openalex_data = openalex_data
            
            # Show summary
            st.info(f"Found {len(openalex_data)} papers from OpenAlex.")
            
            # 3. Build graph
            with st.spinner("Building knowledge graph..."):
                G = set_graph(openalex_data)
                if not G.nodes():
                    st.error("Failed to build graph from the retrieved papers.")
                    st.stop()
                
                st.session_state.G = G
                
                # 4. Compute projections
                with st.spinner("Computing graph projections..."):
                    paper_graph = graph_projection(G, 'paper', 'concept')
                    concept_graph = graph_projection(G, 'concept', 'paper')
                    author_graph = graph_projection(G, 'author', 'paper')
                    
                    st.session_state.projections = {
                        'paper': paper_graph,
                        'concept': concept_graph,
                        'author': author_graph
                    }
                
                # 5. Get concept list for selections
                concept_list = {data.get('label'): node for node, data in G.nodes(data=True) if data.get('node_type') == 'concept'}
                st.session_state.concept_list = concept_list
                
                # 6. Pre-compute analysis based on choice
                choice = user_choice.get('choice')
                graph_figures = {}
                
                if choice == 'idea_evaluation_centrality':
                    with st.spinner("Computing centralities..."):
                        central_papers = compute_centrality(paper_graph)[:10]
                        central_concepts = compute_centrality(concept_graph)[:10]
                        central_authors = compute_centrality(author_graph)[:10]
                        
                        st.session_state.centralities = {
                            'papers': central_papers,
                            'concepts': central_concepts,
                            'authors': central_authors
                        }
                        
                        # Create network visualizations
                        top_paper_nodes = [p['id'] for p in central_papers[:5]]
                        top_concept_nodes = [c['id'] for c in central_concepts[:5]]
                        top_author_nodes = [a['id'] for a in central_authors[:5]]
                        highlighted_nodes = top_paper_nodes + top_concept_nodes + top_author_nodes
                        
                        # Full graph visualization
                        fig_full = create_network_visualization(
                            G, 
                            title="Full Knowledge Graph",
                            layout=graph_layout,
                            show_labels=show_labels,
                            highlight_nodes=highlighted_nodes
                        )
                        graph_figures['full_graph'] = fig_full
                        
                        # Paper graph visualization
                        fig_paper = create_network_visualization(
                            paper_graph,
                            title="Paper Network (Weighted)",
                            layout=graph_layout,
                            show_labels=show_labels,
                            highlight_nodes=top_paper_nodes
                        )
                        graph_figures['paper_graph'] = fig_paper
                        
                        st.session_state.graph_figures = graph_figures
                        
                        # Generate AI interpretation
                        system_prompt = """
                        You are an assistant that will help inform the user on the papers, concepts, and authors that are central to the field that their research inquiry is in.
                        
                        You are to return a message which summarizes the following data (which contains a table of the most central papers, concepts, and authors in the field) so that the researcher is well informed and connects to their inquiry question.
                        
                        The User will give you either the macro-field and microfield OR their research idea:
                        - If they give you the fields, tie your evaluation into a general outlook of the field.
                        - If they give you their research idea, tie your evaluation so that it connects to the research idea.
                        
                        Some information you could give:
                        - What are the most critical papers or concepts that one should know well before writing a paper that discusses the field.
                        - Are there certain authors to be noted for.
                        - How do these help or connect with the research idea that the user is asking (in case 2).
                        
                        Note: Ensure that you adapt the information given to you to answer the inquiry of the user (also specific to their research idea as well if that is the case).
                        """
                        
                        user_prompt = f"""
                        Inquiry: {inq}
                        Research idea: {research_idea}
                        Macro-field: {macro}
                        Micro-field: {micro}
                        This is the data on the most central papers:
                        {json.dumps(central_papers[:5])}
                        This is the data on the most central concepts:
                        {json.dumps(central_concepts[:5])}
                        This is the data on the most central authors:
                        {json.dumps(central_authors[:5])}
                        """
                        
                        final_response = ask_deepseek(system_prompt, user_prompt)
                        st.session_state.ai_response = final_response
                
                elif choice == 'concept_shortest_path':
                    if not concept_list:
                        st.warning("No concepts found in the retrieved papers.")
                    else:
                        # Create full graph visualization
                        fig_full = create_network_visualization(
                            G,
                            title="Full Knowledge Graph",
                            layout=graph_layout,
                            show_labels=show_labels
                        )
                        graph_figures['full_graph'] = fig_full
                        
                        # Create concept graph visualization
                        fig_concept = create_network_visualization(
                            concept_graph,
                            title="Concept Network",
                            layout=graph_layout,
                            show_labels=show_labels
                        )
                        graph_figures['concept_graph'] = fig_concept
                        
                        st.session_state.graph_figures = graph_figures
                
                elif choice == 'paper_community_detection':
                    with st.spinner("Detecting communities..."):
                        paper_communities = community_detection(paper_graph)
                        concept_communities = community_detection(concept_graph)
                        
                        st.session_state.communities = {
                            'papers': paper_communities,
                            'concepts': concept_communities
                        }
                        
                        # Create community visualizations
                        # First, add community info to graphs for visualization
                        for community_data in paper_communities:
                            node = community_data['id']
                            community = community_data['community']
                            if paper_graph.has_node(node):
                                paper_graph.nodes[node]['community'] = community
                        
                        for community_data in concept_communities:
                            node = community_data['id']
                            community = community_data['community']
                            if concept_graph.has_node(node):
                                concept_graph.nodes[node]['community'] = community
                        
                        # Create community visualizations
                        fig_paper_community = create_community_visualization(
                            paper_graph,
                            layout=graph_layout,
                            show_labels=show_labels
                        )
                        graph_figures['paper_community'] = fig_paper_community
                        
                        fig_concept_community = create_community_visualization(
                            concept_graph,
                            layout=graph_layout,
                            show_labels=show_labels
                        )
                        graph_figures['concept_community'] = fig_concept_community
                        
                        # Full graph visualization
                        fig_full = create_network_visualization(
                            G,
                            title="Full Knowledge Graph",
                            layout=graph_layout,
                            show_labels=show_labels
                        )
                        graph_figures['full_graph'] = fig_full
                        
                        st.session_state.graph_figures = graph_figures
                        
                        # Generate AI interpretation
                        system_prompt = """
                        You are an assistant that will help inform the user on the papers and concept communities (or subfields) that are central to the field that their research inquiry is in.
                        
                        You are to return a message which summarizes the following data (which contains a table of the most central papers and concept subfields in the field) so that the researcher is well informed and connects to their inquiry question.
                        
                        The User will give you either the macro-field and microfield OR their research idea:
                        - If they give you the fields, tie your evaluation into a general outlook of the field.
                        - If they give you their research idea, tie your evaluation so that it connects to the research idea.
                        
                        Some information you could give:
                        - What are the most critical concepts that papers are focusing on that one should know well before writing a paper that discusses the field.
                        - Are there certain authors to be noted for.
                        - How do these help or connect with the research idea that the user is asking (in case 2).
                        
                        IMPORTANT: Based on the inquiry question of the user, determine which community (of paper or concept or both) to use for evaluation to answer the user.
                        
                        Note: Ensure that you adapt the information given to you to answer the inquiry of the user.
                        """
                        
                        user_prompt = f"""
                        Inquiry: {inq}
                        Macro-field: {macro}
                        Micro-field: {micro}
                        Research idea: {research_idea}
                        This is the data on the communities of papers:
                        {json.dumps(paper_communities[:20])}
                        This is the data on the communities of concepts:
                        {json.dumps(concept_communities[:20])}
                        """
                        
                        final_response = ask_deepseek(system_prompt, user_prompt)
                        st.session_state.ai_response = final_response
        
        st.session_state.analysis_complete = True
        st.session_state.step = "results"
        st.rerun()

# -------------------------------------------------------------------
# Results Display
# -------------------------------------------------------------------

if st.session_state.step == "results" and st.session_state.analysis_complete:
    choice = st.session_state.user_choice.get('choice')
    G = st.session_state.G
    projections = st.session_state.projections
    user_choice = st.session_state.user_choice
    concept_list = st.session_state.concept_list
    graph_figures = st.session_state.graph_figures or {}
    
    st.divider()
    st.header("📊 Analysis Results")
    
    # Display the papers found
    with st.expander(f"📄 Papers Found ({len(st.session_state.openalex_data)})"):
        papers_df = pd.DataFrame(st.session_state.openalex_data)
        display_cols = ['display_name', 'publication_year', 'cited_by_count']
        available_cols = [col for col in display_cols if col in papers_df.columns]
        if available_cols:
            st.dataframe(papers_df[available_cols], use_container_width=True)
        else:
            st.write("No paper details available.")
    
    # Analysis based on choice
    if choice == 'idea_evaluation_centrality':
        st.subheader("🎯 Centrality Analysis")
        
        central_papers = st.session_state.centralities.get('papers', [])
        central_concepts = st.session_state.centralities.get('concepts', [])
        central_authors = st.session_state.centralities.get('authors', [])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Top Papers**")
            if central_papers:
                for i, paper in enumerate(central_papers, 1):
                    st.write(f"{i}. {paper.get('label', 'Unknown')} (Score: {paper.get('connection_score', 0):.1f})")
            else:
                st.write("No papers found.")
                
        with col2:
            st.write("**Top Concepts**")
            if central_concepts:
                for i, concept in enumerate(central_concepts, 1):
                    st.write(f"{i}. {concept.get('label', 'Unknown')} (Score: {concept.get('connection_score', 0):.1f})")
            else:
                st.write("No concepts found.")
                
        with col3:
            st.write("**Top Authors**")
            if central_authors:
                for i, author in enumerate(central_authors, 1):
                    st.write(f"{i}. {author.get('label', 'Unknown')} (Score: {author.get('connection_score', 0):.1f})")
            else:
                st.write("No authors found.")
        
        # Display graph visualizations
        st.subheader("📊 Network Visualizations")
        
        if graph_figures.get('full_graph'):
            st.write("**Full Knowledge Graph**")
            st.pyplot(graph_figures['full_graph'])
        
        if graph_figures.get('paper_graph'):
            st.write("**Paper Network (Weighted Projection)**")
            st.pyplot(graph_figures['paper_graph'])
        
        # AI Interpretation
        if st.session_state.ai_response:
            st.subheader("🧠 AI Interpretation")
            st.write(st.session_state.ai_response)
    
    elif choice == 'concept_shortest_path':
        st.subheader("🔗 Shortest Path Analysis")
        
        # Display graph visualizations
        st.subheader("📊 Network Visualizations")
        
        if graph_figures.get('full_graph'):
            st.write("**Full Knowledge Graph**")
            st.pyplot(graph_figures['full_graph'])
        
        if graph_figures.get('concept_graph'):
            st.write("**Concept Network**")
            st.pyplot(graph_figures['concept_graph'])
        
        if not concept_list:
            st.warning("No concepts found in the retrieved papers.")
        else:
            conceptA_label = st.selectbox("Select first concept", list(concept_list.keys()), key="conceptA")
            remaining_concepts = [k for k in concept_list.keys() if k != conceptA_label]
            
            if remaining_concepts:
                conceptB_label = st.selectbox("Select second concept", remaining_concepts, key="conceptB")
                
                if st.button("🔍 Find Path", type="primary", key="find_path"):
                    conceptA = concept_list[conceptA_label]
                    conceptB = concept_list[conceptB_label]
                    
                    with st.spinner("Finding shortest path..."):
                        path_data = shortest_path(projections['concept'], conceptA, conceptB)
                        st.session_state.path_data = path_data
                        st.rerun()
                
                # Display path if it exists in session state
                if st.session_state.path_data:
                    path_data = st.session_state.path_data
                    
                    if not path_data:
                        st.warning("No path found between these two concepts.")
                    else:
                        st.write(f"**Path length: {len(path_data)} steps**")
                        
                        # Display path
                        for step in path_data:
                            with st.container():
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.write(f"**Step {step['step']}**")
                                with col2:
                                    st.write(f"**{step.get('label', 'Unknown')}**")
                                    if step.get('year'):
                                        st.write(f"Year: {step['year']}")
                                    if step.get('abstract'):
                                        st.write(f"Abstract: {step['abstract'][:200]}...")
                                st.divider()
                        
                        # Create path visualization
                        if concept_graph and path_data:
                            path_fig = create_path_visualization(
                                concept_graph,
                                path_data,
                                layout=graph_layout,
                                show_labels=show_labels
                            )
                            if path_fig:
                                st.subheader("🗺️ Path Visualization")
                                st.pyplot(path_fig)
                        
                        # Generate AI interpretation for the path
                        if st.button("Generate AI Interpretation for Path", key="generate_path_ai"):
                            with st.spinner("Generating AI analysis..."):
                                system_prompt = """
                                You are an assistant that will help inform the user on how two concepts are connected to each other.
                                
                                You are to return a message which summarizes the following data (which contains a table of the steps of connections between the two concepts) so that the researcher is well informed and connects to their inquiry question.
                                
                                The User will give you either the macro-field and microfield OR their research idea:
                                - If they give you the fields, tie your evaluation into a general outlook of the field.
                                - If they give you their research idea, tie your evaluation so that it connects to the research idea.
                                
                                Some information you could give:
                                - Explanation on each step of the connection from the two concepts.
                                - How do these help or connect with the research idea that the user is asking (in case 2).
                                
                                Note: Ensure that you adapt the information given to you to answer the inquiry of the user.
                                """
                                
                                user_prompt = f"""
                                Inquiry: {inq}
                                Macro-field: {macro}
                                Micro-field: {micro}
                                Research idea: {research_idea}
                                This is the data on the steps of connections between {conceptA_label} and {conceptB_label}:
                                {json.dumps(path_data)}
                                """
                                
                                final_response = ask_deepseek(system_prompt, user_prompt)
                                if final_response:
                                    st.subheader("🧠 AI Interpretation")
                                    st.write(final_response)
            else:
                st.warning("Only one concept available. Please adjust your search to find more concepts.")
    
    elif choice == 'paper_community_detection':
        st.subheader("👥 Community Detection Analysis")
        
        paper_communities = st.session_state.communities.get('papers', [])
        concept_communities = st.session_state.communities.get('concepts', [])
        
        # Display graph visualizations
        st.subheader("📊 Network Visualizations")
        
        if graph_figures.get('full_graph'):
            st.write("**Full Knowledge Graph**")
            st.pyplot(graph_figures['full_graph'])
        
        col1, col2 = st.columns(2)
        with col1:
            if graph_figures.get('paper_community'):
                st.write("**Paper Communities**")
                st.pyplot(graph_figures['paper_community'])
        
        with col2:
            if graph_figures.get('concept_community'):
                st.write("**Concept Communities**")
                st.pyplot(graph_figures['concept_community'])
        
        # Display communities summary
        if paper_communities:
            df_papers = pd.DataFrame(paper_communities)
            communities = df_papers['community'].unique()
            
            st.write(f"**Found {len(communities)} paper communities**")
            
            # Show community sizes
            community_sizes = df_papers.groupby('community').size().sort_values(ascending=False)
            st.bar_chart(community_sizes)
            
            # Show top papers per community
            st.write("**Top papers per community:**")
            for community in sorted(communities):
                community_papers = df_papers[df_papers['community'] == community].sort_values('citation_count', ascending=False).head(3)
                with st.expander(f"Community {community} ({len(df_papers[df_papers['community'] == community])} papers)"):
                    for _, paper in community_papers.iterrows():
                        st.write(f"• {paper['label']} (Citations: {paper.get('citation_count', 0)})")
                        if paper.get('abstract'):
                            st.write(f"  {paper['abstract'][:150]}...")
        else:
            st.write("No paper communities found.")
        
        # Display concept communities
        if concept_communities:
            st.write("**Concept Communities**")
            df_concepts = pd.DataFrame(concept_communities)
            concept_communities_found = df_concepts['community'].unique()
            
            for community in sorted(concept_communities_found):
                community_concepts = df_concepts[df_concepts['community'] == community]
                with st.expander(f"Concept Community {community} ({len(community_concepts)} concepts)"):
                    for _, concept in community_concepts.head(10).iterrows():
                        st.write(f"• {concept['label']}")
        
        # AI Interpretation
        if st.session_state.ai_response:
            st.subheader("🧠 AI Interpretation")
            st.write(st.session_state.ai_response)
    
    else:
        st.error("Unknown analysis choice. Please run the analysis again.")
    
    # Reset button
    if st.button("🔄 Start New Analysis", type="secondary"):
        for key in ['step', 'results', 'G', 'projections', 'user_choice', 'openalex_data', 
                   'concept_list', 'centralities', 'communities', 'path_data', 'ai_response', 
                   'analysis_complete', 'graph_figures']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Footer
st.divider()
st.caption("AI-Powered Research Explorer v2.0 | Powered by OpenAlex and DeepSeek")
