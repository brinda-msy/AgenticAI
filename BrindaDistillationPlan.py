import os
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Load environment variables and initialize OpenAI client
load_dotenv()
client = OpenAI(
base_url="https://openai.vocareum.com/v1",
api_key=os.getenv("OPENAI_API_KEY"))


def call_openai(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    """Simple wrapper for OpenAI API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def feedstock_analyst_agent(feedstock_name):
    """Analyze the hydrocarbon feedstock."""

    print("\n--- Agent 1: Feedstock Analyst ---")
    print(f"Analyzing feedstock: {feedstock_name}")

    system_prompt = """
    You are a petrochemical expert analyzing hydrocarbon feedstocks.
    Provide a concise analysis of the given feedstock, highlighting:
    - likely hydrocarbon components
    - general characteristics
    - suitability for producing valuable refined products
      such as gasoline, diesel, and kerosene.

    Keep the analysis practical and concise.
    """

    user_prompt = f"Analyze the feedstock: {feedstock_name}"

    response = call_openai(system_prompt, user_prompt)

    print("Feedstock analysis complete.")

    return response


def distillation_planner_agent(feedstock_analysis):
    """Estimate product yields based on feedstock analysis."""

    print("\n--- Agent 2: Distillation Planner ---")
    print("Creating a potential product yield plan...")

    system_prompt = """
    You are a refinery distillation tower operations planner.

    Based on the provided feedstock analysis, estimate potential
    percentage yields for major refined products such as gasoline,
    diesel, and kerosene.

    Be realistic and clearly state that these are estimates rather
    than actual refinery operating yields.

    Ensure the estimated percentages add up to approximately 100%.
    """

    user_prompt = f"""
    Based on the following feedstock analysis, create a potential
    distillation product yield plan.

    Feedstock analysis:
    {feedstock_analysis}
    """

    response = call_openai(system_prompt, user_prompt)

    print("Distillation plan complete.")

    return response


def extract_product_list(distillation_plan):
    """
    Ask the LLM to extract the major product names from the
    distillation plan.
    """

    system_prompt = """
    You are a data extraction assistant.

    Extract only the names of the major refined products mentioned
    in the provided distillation plan.

    Return them as a simple comma-separated list.
    Do not include percentages, explanations, or other text.
    """

    user_prompt = f"""
    Extract the refined product names from this distillation plan:

    {distillation_plan}
    """

    return call_openai(system_prompt, user_prompt)


def market_analyst_agent(product_list):
    """Analyze market conditions for refined products."""

    print("\n--- Agent 3: Market Analyst ---")
    print(f"Analyzing market for: {product_list}")

    system_prompt = """
    You are an energy market analyst.

    For the following list of refined products, provide a brief
    analysis of:
    - current/general market demand
    - demand level: high, medium, or low
    - general profitability trends
    - any important market considerations

    Do not invent precise current prices. Focus on general market
    trends and relative profitability.
    """

    user_prompt = f"""
    Analyze the market for these refined products:

    {product_list}
    """

    response = call_openai(system_prompt, user_prompt)

    print("Market analysis complete.")

    return response


def production_optimizer_agent(distillation_plan, market_data):
    """Recommend an optimized production strategy."""

    print("\n--- Agent 4: Production Optimizer ---")
    print("Combining yield and market information...")

    system_prompt = """
    You are a refinery production optimization expert.

    Your goal is to recommend a production strategy based on:
    1. Potential refinery yields
    2. Market demand
    3. General profitability

    Compare the products and recommend which products should receive
    production focus.

    Clearly explain the reasoning behind the recommendation.

    Remember that this is a high-level planning recommendation and
    not a substitute for actual refinery engineering, safety,
    regulatory, or operational decisions.
    """

    user_prompt = f"""
    Create a recommended production strategy using the information
    below.

    DISTILLATION PLAN:
    {distillation_plan}

    MARKET DATA:
    {market_data}

    Provide a concise final recommendation that balances refinery
    yield potential with market demand and profitability.
    """

    response = call_openai(system_prompt, user_prompt)

    print("Production optimization complete.")

    return response


if __name__ == "__main__":

    # Initial feedstock
    current_feedstock = "West Texas Intermediate Crude"

    print("=" * 60)
    print("REFINERY OPTIMIZATION AGENT CHAIN")
    print("=" * 60)

    # Agent 1
    feedstock_analysis = feedstock_analyst_agent(
        current_feedstock
    )

    print("\nFeedstock Analysis:")
    print(feedstock_analysis)

    # Agent 2
    distillation_plan = distillation_planner_agent(
        feedstock_analysis
    )

    print("\nDistillation Plan:")
    print(distillation_plan)

    # Extract products from Agent 2 output
    product_list = extract_product_list(
        distillation_plan
    )

    print("\nExtracted Product List:")
    print(product_list)

    # Agent 3
    market_data = market_analyst_agent(
        product_list
    )

    print("\nMarket Analysis:")
    print(market_data)

    # Agent 4
    final_recommendation = production_optimizer_agent(
        distillation_plan,
        market_data
    )

    print("\nFinal Production Recommendation:")
    print(final_recommendation)

    print("\n" + "=" * 60)
    print("REFINERY OPTIMIZATION COMPLETE")
    print("=" * 60)