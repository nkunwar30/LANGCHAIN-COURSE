from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


MAX_Iteration = 10
MODEL = "qwen3:1.7b"

# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"  >>> Executing get_product_price for product: {product}")
    # Simulate getting the price of a product
    prices = {
        "laptop": "$1200",
        "smartphone": "$499",
        "headphones": "$199",
        "keyboard": "$99",
    }
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers: bronze, silver, gold."""
    print(
        f"  >>> Executing apply_discount(price: {price},discount_tier: {discount_tier})"
    )
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 15}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content="You are a helpful shopping assistant."
            "You have access to a product catalog tool "
            " and a discount tool. \n\m"
            "STRICT RULES - you must follow these exactly:\n"
            "1. NEVER guess or assume any product price."
            "You must call get_product_price first to get the real price.\n"
            "2. Only call apply_discount after you have received a price from get_product_price. "
            "Pass the exact price you received from get_product_price - do NOT pass a made-up number.\n"
            "3. NEVER calculate discounts yourself using math. Always use the apply_discount ttol.\n"
            "4. If user does not specify a discount tier, ask them which tier to use - DO NOT assume one "
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_Iteration + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        # if no tool calls, this is the final answer

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f" [Tool Selected: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Error: Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)

        print(f" [Tool Result]: {observation}]")

        messages.append(
            ai_message
        )  # Add the AI message with tool call to the conversation
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )  # Add the tool result as a ToolMessage

    print("Error: Max iterations reached without a final answer.")
    return None


if __name__ == "__main__":  # Example question that requires tool calls
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
