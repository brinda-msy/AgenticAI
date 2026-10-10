import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))

MAX_RETRIES = 5

# Example user constraints
RECIPE_REQUEST = {
    "base_dish": "pasta",
    "constraints": [
        "gluten-free",
        "vegan",
        "under 500 calories per serving",
        "high protein (>15g per serving)",
        "no coconut",
        "taste must be rated 7/10 or higher"
    ]
}

print("API key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Key length:", len(os.getenv("OPENAI_API_KEY", "")))

class RecipeCreatorAgent:
    def create_recipe(self, recipe_request_dict, feedback=None):
        system_prompt = """
        You are an innovative and highly skilled chef who specializes in
        creating delicious recipes that meet specific dietary and nutritional
        requirements.

        You are creative, adaptable, and precise. When given feedback about
        a previous recipe, revise the recipe to address that feedback while
        continuing to satisfy all of the original constraints.

        Provide recipes with:
        - A creative dish name
        - Ingredients with quantities
        - Step-by-step instructions
        - Estimated calories per serving
        - Estimated protein in grams per serving
        - A brief taste profile
        """

        base_dish = recipe_request_dict["base_dish"]
        constraints = recipe_request_dict["constraints"]
        constraints_str = ", ".join(constraints)

        user_prompt = f"""
        Please create a recipe for "{base_dish}" that meets ALL of the
        following constraints:

        {constraints_str}
        """

        if feedback:
            user_prompt += f"""

        IMPORTANT: This is a revision of a previous recipe.

        The previous recipe was evaluated and received the following feedback:

        {feedback}

        Revise the recipe to address all of this feedback while continuing
        to satisfy ALL of the original constraints.
        """
        else:
            user_prompt += """

        This is the first attempt at creating the recipe.
        """

        user_prompt += """

        Please provide:
        1. A creative name for the dish
        2. A list of ingredients with quantities
        3. Step-by-step cooking instructions
        4. Estimated calories per serving
        5. Estimated protein in grams per serving
        6. A short description of the taste profile
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8
        )

        recipe = response.choices[0].message.content

        print("\n👨‍🍳 Chef's Recipe:")
        print(recipe)

        return recipe
       
class NutritionEvaluatorAgent:
    def evaluate(self, recipe_details_str, original_request_dict):

        system_prompt = """
        You are an extremely precise and meticulous nutritionist and food
        critic.
        

        Your job is to evaluate a recipe against a specific set of
        user-defined dietary, nutritional, ingredient, and taste constraints.

        For EVERY constraint:
        - State the constraint exactly as provided.
        - Mark it as PASSED or FAILED.
        - If it FAILED, provide a concise reason and a specific,
          actionable suggestion for improvement.

        You must be strict and objective. Do not assume that a recipe meets
        a constraint simply because the chef claims that it does.

        You must also provide a Taste Rating from 1 to 10 based on the
        recipe's described taste profile.

        Finally, provide an Overall Status:
        - PASSED only if ALL constraints are satisfied.
        - FAILED if ANY constraint is not satisfied.

        Use the exact output format requested by the user so that another
        program can reliably parse your evaluation.
        For numerical constraints, evaluate the stated numerical values literally.
        Do not fail a constraint based on hypothetical or speculative possibilities.

        For example:
        - 480 calories satisfies "under 500 calories per serving".
        - 20g protein satisfies "high protein (>15g per serving)".
        - A taste rating of 8/10 satisfies "taste must be rated 7/10 or higher".

        Only mark a constraint FAILED when the recipe clearly violates that
        constraint based on the information provided.
        """

        constraints = original_request_dict["constraints"]
        constraints_str = ", ".join(constraints)

        user_prompt = f"""
        Please evaluate the following RECIPE against EACH constraint from
        the original request.

        RECIPE:
        {recipe_details_str}

        ORIGINAL REQUEST CONSTRAINTS:
        {constraints_str}

        For each constraint, state the constraint verbatim followed by
        PASSED or FAILED.

        If a constraint FAILED, provide:
        - A brief reason explaining why it failed.
        - A specific suggestion for fixing it.

        Use this format:

        gluten-free: PASSED
        vegan: PASSED
        under 500 calories per serving: FAILED - Estimated 550 calories.
        Suggestion: Reduce the amount of olive oil.

        After evaluating every constraint, provide:

        Taste Rating: [N]/10

        Finally, provide:

        Overall Status: PASSED

        ONLY use Overall Status: PASSED if ALL constraints are satisfied,
        including the requirement that the taste rating must be 7/10 or
        higher.

        Otherwise use:

        Overall Status: FAILED
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        evaluation = response.choices[0].message.content

        print("\n🧐 Critic's Evaluation:")
        print(evaluation)

        return evaluation    

#if __name__ == "__main__":
    #recipe_creator = RecipeCreatorAgent()
    #evaluator = NutritionEvaluatorAgent()

    #recipe = recipe_creator.create_recipe(RECIPE_REQUEST)

    #evaluation = evaluator.evaluate(
        #recipe,
        #RECIPE_REQUEST
    #)        

def optimize_recipe():
    current_feedback = None

    recipe_creator = RecipeCreatorAgent()
    evaluator = NutritionEvaluatorAgent()

    current_recipe_str = None
    evaluation_str = None

    for attempt in range(MAX_RETRIES):
        print(f"\n--- Attempt {attempt + 1} of {MAX_RETRIES} ---")

        # Ask the chef to create or revise the recipe
        current_recipe_str = recipe_creator.create_recipe(
            RECIPE_REQUEST,
            current_feedback
        )

        print(f"\n💡 Chef Proposed:\n{current_recipe_str}")

        # Ask the critic to evaluate the recipe
        evaluation_str = evaluator.evaluate(
            current_recipe_str,
            RECIPE_REQUEST
        )

        print(f"\n🧐 Critic's Evaluation:\n{evaluation_str}")

        # Check whether the recipe passed all constraints
        if "overall status: passed" in evaluation_str.lower():
            print("\n✅ Recipe Approved by Evaluator!")

            return (
                current_recipe_str,
                evaluation_str,
                attempt + 1
            )

        # Recipe failed, so send the evaluation back to the chef
        current_feedback = evaluation_str

        print("\nRecipe needs revision. Relaying feedback to Chef...")

    # No recipe passed within MAX_RETRIES
    print(
        f"\n❌ Failed to meet all constraints after "
        f"{MAX_RETRIES} attempts."
    )

    return (
        current_recipe_str,
        evaluation_str,
        MAX_RETRIES
    )


if __name__ == "__main__":
    print("Starting AI Recipe Optimizer Workflow...")

    final_recipe, final_evaluation, attempts = optimize_recipe()

    print("\n========== FINAL RESULT ==========")
    print(f"Approved after {attempts} attempt(s).")

    print("\nFinal Recipe:")
    print(final_recipe)

    print("\nFinal Evaluation:")
    print(final_evaluation)