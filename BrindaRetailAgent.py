import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- Helper Function for API Calls ---
def call_openai(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    """Simple wrapper for OpenAI API calls."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# --- Agents for Different Retail Tasks ---

def product_researcher_agent(query):
    """Product researcher agent gathers product information."""
    system_prompt = """You are a product research agent for a retail company. Your task is to provide 
    structured information about products, market trends, and competitor pricing."""
    
    user_prompt = f"Research this product thoroughly: {query}"
    return call_openai(system_prompt, user_prompt)


def customer_analyzer_agent(query):
    """Customer analyzer agent processes customer data and feedback."""
    system_prompt = """You are a customer analysis agent. Your task is to analyze customer feedback, 
    preferences, and purchasing patterns."""
    
    user_prompt = f"Analyze customer behavior for: {query}"
    return call_openai(system_prompt, user_prompt)


def pricing_strategist_agent(query, product_data=None, customer_data=None):
    """Pricing strategist agent recommends optimal pricing."""
    
    system_prompt = """You are an expert retail pricing strategist. Your goal is to recommend an optimal 
    pricing strategy and price point based on detailed product research and customer analysis."""
    
    user_prompt = f"""
    Original Pricing Query:
    {query}

    Product Research Data:
    {product_data}

    Customer Analysis Data:
    {customer_data}

    Based on all the above information, please provide a recommended pricing strategy, 
    suggest an optimal price or price range, and explain your reasoning.
    """
    
    print("Pricing Strategist Agent is analyzing the gathered information...")
    
    return call_openai(system_prompt, user_prompt)


# --- Routing Agent with LLM-Based Task Determination ---
def routing_agent(query, *args):
    """Routing agent that determines which agent to use based on the query."""
    
    # Step 1: Classify the query using an LLM
    classification_system_prompt = """You are a helpful AI assistant that categorizes 
    retail-related user queries.

    Based on the user's query, determine if it is primarily about:

    - "product research" (e.g., asking for product specs, trends, competitor prices)
    - "customer analysis" (e.g., asking about customer feedback, preferences, purchase patterns)
    - "pricing strategy" (e.g., asking for optimal pricing for a product)

    Respond only with one of these exact phrases:
    "product research", "customer analysis", or "pricing strategy".
    """
    
    classification_user_prompt = f"Categorize this query: {query}"
    
    task_type = call_openai(
        classification_system_prompt,
        classification_user_prompt
    ).strip()
    
    print(f"Task classified as: {task_type}")
    
    # Step 2: Route based on the task type
    
    if task_type == "product research":
        print("Routing to Product Researcher Agent...")
        return product_researcher_agent(query)
    
    elif task_type == "customer analysis":
        print("Routing to Customer Analyzer Agent...")
        return customer_analyzer_agent(query)
    
    elif task_type == "pricing strategy":
        print("Pricing strategy query identified. Gathering prerequisite information...")
        
        # First gather product research
        print("  -> Calling Product Researcher Agent...")
        product_info = product_researcher_agent(query)
        
        # Then gather customer analysis
        print("  -> Calling Customer Analyzer Agent...")
        customer_info = customer_analyzer_agent(query)
        
        # Finally send both results to the pricing strategist
        print("  -> Calling Pricing Strategist Agent...")
        return pricing_strategist_agent(
            query,
            product_info,
            customer_info
        )
    
    else:
        print(f"Could not determine appropriate agent for query: {query}")
        return "Sorry, I could not understand how to route your request."


# --- Example Usage ---
if __name__ == "__main__":
    
    # Example queries
    queries = [
        "What are the specifications and current market trends for wireless earbuds?",
        "What do customers think about our premium coffee brand?",
        "What should be the optimal price for our new organic skincare line?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("\nProcessing...")
        
        result = routing_agent(query)
        
        print("\n--- FINAL RESULT ---")
        print(result)
        print("=" * 50)