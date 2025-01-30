import os
import requests
from typing import Dict, Any
from langsmith import traceable

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content in the formatted string.
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

@traceable
def searxng_search(query, include_raw_content=True, max_results=3):
    """Search the web using a local SearxNG instance.
    
    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw content in the formatted string
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """
    # Configure your local SearxNG instance URL
    SEARXNG_URL = os.getenv('SEARXNG_URL', 'http://localhost:8080')
    
    # Set up the search parameters
    params = {
        'q': query,
        'format': 'json',
        'engines': 'google,bing,duckduckgo',  # Customize engines as needed
        'language': 'en',
        'max_results': max_results
    }
    
    try:
        response = requests.get(f"{SEARXNG_URL}/search", params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Transform SearxNG results to match the expected format
        formatted_results = []
        for result in search_results.get('results', [])[:max_results]:
            formatted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'content': result.get('content', ''),
                'raw_content': None  # SearxNG doesn't provide full content by default
            }
            
            # Optionally fetch raw content if requested
            if include_raw_content:
                try:
                    content_response = requests.get(formatted_result['url'], timeout=5)
                    if content_response.status_code == 200:
                        formatted_result['raw_content'] = content_response.text
                except Exception as e:
                    print(f"Failed to fetch raw content for {formatted_result['url']}: {str(e)}")
            
            formatted_results.append(formatted_result)
            
        return {'results': formatted_results}
        
    except requests.exceptions.RequestException as e:
        print(f"Error searching SearxNG: {str(e)}")
        return {'results': []}

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API."""
    # Perplexity search implementation remains unchanged
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations", ["https://perplexity.ai"])
    
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}