from autogen import AssistantAgent, UserProxyAgent
from utils import get_config
#from call_uipath_process import call_uipath_process as base_call_uipath_process
#use this for commnand line execution
from .call_uipath_process import call_uipath_process as base_call_uipath_process
from typing import Optional, Dict, Any

config, api_key, endpoint, apiversion, model = get_config()

llm_config = {
    "temperature": 0,
    "config_list": config,
    "timeout": 120
}

INTENT_TO_PROCESS = {
    "generate product info": "ProductInfoAPI",
    "extract document text": "PDF.to.Text._RPA_",
    "validate invoice" : "Invoice.Contract.Validation._Orchestration_",
    "create invoice": "Invoice.Demo.Processing"
}

INTENT_KEYWORDS = {
    "generate product info": ["product", "info", "generate", "product info", "payments", "outstanding"],
    "extract document text": ["extract", "document", "text", "extraction", "parse", "read"],
    "validate invoice": ["validate", "invoice", "check", "contract", "validation","verify"],
    "create invoice": ["create", "invoice", "generate", "process"]
}

INTENT_INPUT_ARGUMENTS = {
    "extract document text": {
        "in_StorageBucket": "Contracts",
        "in_FileName" :"Contract_Services_Agreement.pdf"},
    "generate product info": {},
    "validate invoice":{},
    "create invoice": {
        "in_ClientName": "Microsoft",
        "in_InvoiceDate": "08/20/2025",
        "in_InvoiceDueDate": "09/20/2025",
        "in_RateAmount": 1000.00,
        "in_TotalHours": 10,
        "in_LineItemDescription": "Demo invoice for services rendered",
        "In_Services": "Consulting Services",
        "in_InvoiceNumber": "INV-0012345",
        "in_Service": True,
        "in_PO_Number": "PO-INV-00123"
    }
}

def find_best_intent_match(user_input: str) -> Optional[str]:
    """
    Find the best matching intent based on user input using keyword matching.
    
    Args:
        user_input: The user's natural language input
    
    Returns:
        The best matching intent string or None if no good match found
    """
    user_input_lower = user_input.lower().strip()
    
    # First try exact match
    if user_input_lower in INTENT_TO_PROCESS:
        return user_input_lower
    
    # Then try keyword matching
    best_match = None
    best_score = 0
    
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in user_input_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = intent
    
    # Only return a match if we found at least one keyword
    return best_match if best_score > 0 else None

def call_uipath_process(intent: str, input_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    #Trigger a UiPath RPA process based on intent.

    # Debug: Print what we're looking for
    print(f"Original user input: '{intent}'")
    
    # Try to find the best matching intent
    matched_intent = find_best_intent_match(intent)
    
    print(f"Matched intent: '{matched_intent}'")
    print(f"Available intents: {list(INTENT_TO_PROCESS.keys())}")
    
    if not matched_intent:
        error_msg = f"Could not match intent for: '{intent}'. Available intents: {list(INTENT_TO_PROCESS.keys())}"
        print(error_msg)
        return {"status": "error", "message": error_msg}
    
    # Get the process name
    process_name = INTENT_TO_PROCESS.get(matched_intent)
    
    # Debug: Confirm the process name found
    print(f"Found process name: '{process_name}'")
    
    # Get input arguments
    inputs = input_args if input_args is not None else INTENT_INPUT_ARGUMENTS.get(matched_intent, {})
    
    print(f"Using inputs: {inputs}")

    try:
        result = base_call_uipath_process(process_name, inputs)
        return result
    except Exception as e:
        error_msg = f"Error calling UiPath process '{process_name}': {e}"
        return {"status": "error", "message": error_msg}

def run_uipath_agent():
    # Create the assistant agent
    assistant = AssistantAgent(
        name="UiPathAgent",
        llm_config=llm_config,
        system_message=(
            "You are an automation agent that helps trigger UiPath RPA processes.\n"
            "When a user requests automation tasks, analyze their request and call the 'call_uipath_process' function.\n"
            "Available processes:\n"
            "1. For generating product information or checking payments: use 'generate product info'\n"
            "2. For extracting text from documents: use 'extract document text'\n"
            "\nIMPORTANT: You must call the function with the user's original request text, not just the intent.\n"
            "The system will automatically match the intent. For example:\n"
            "- User says: 'I need to generate product information' -> call_uipath_process('I need to generate product information')\n"
            "- User says: 'extract text from document' -> call_uipath_process('extract text from document')\n"
            "- User says: 'check for outstanding payments' -> call_uipath_process('check for outstanding payments')\n"
            "Always respond with TERMINATE after successfully calling the function."
        )
    )

    # Create the user proxy agent
    user = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",  # Changed from "ALWAYS" to allow automatic execution
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": ".coding",
            "use_docker": False,
        }
    )

    # Register the function with the user proxy agent
    user.register_function(
        function_map={
            "call_uipath_process": call_uipath_process
        }
    )

    # Start the conversation
    chat_result = user.initiate_chat(
        assistant,
        message="generate product info",
        silent=False  # Set to True if you want less verbose output
    )
    
    return chat_result

# Alternative version with human input if needed
def run_uipath_agent_with_human_input():
    # Create a wrapper function to capture the exact user input
    def process_user_request(user_message: str) -> Dict[str, Any]:
        """
        Process the user's request and call UiPath with their exact message.
        """
        print(f"DEBUG: User's exact message: '{user_message}'")
        result = call_uipath_process(user_message)
        print(f"DEBUG: Function result: {result}")
        return result

    assistant = AssistantAgent(
        name="UiPathAgent",
        llm_config=llm_config,
        system_message=(
            "You are an automation agent that helps trigger UiPath RPA processes.\n"
            "When a user describes what they need, you should:\n"
            "1. Acknowledge their request\n"
            "2. Call process_user_request with their EXACT message text\n"
            "3. Report the result back to the user\n"
            "4. End with TERMINATE\n"
            "\nAvailable processes:\n"
            "- Product information and payment checking\n"
            "- Document text extraction\n"
            "\nIMPORTANT: Always pass the user's complete original message to process_user_request().\n"
            "For example:\n"
            "User: 'I need to extract text from a document'\n"
            "You: 'I'll help you extract text from a document.'\n"
            "Then call: process_user_request('I need to extract text from a document')\n"
            "Never modify or interpret the user's request - pass it exactly as they wrote it."
        )
    )

    user = UserProxyAgent(
        name="User",
        human_input_mode="ALWAYS",  # Human input enabled
        max_consecutive_auto_reply=3,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": ".coding",
            "use_docker": False,
        }
    )

    # Register the wrapper function instead
    user.register_function(
        function_map={
            "process_user_request": process_user_request
        }
    )

    # Custom message handler to ensure we capture user input properly
    def custom_chat_handler():
        print("Hello! I'm your UiPath automation assistant. I can help you with:")
        print("- Generating product information and checking payments")
        print("- Extracting text from documents")
        print("- Compare Invoice against contract agreement")
        print("\nWhat automation task would you like me to help you with today?")
        print("Please describe what you need in your own words.")
        
        # Get the user's actual request
        user_input = input("\nYour request: ").strip()
        
        if not user_input:
            print("No input received.")
            return {"status": "error", "message": "No input received"}
        
        print(f"\nProcessing your request: '{user_input}'")
        
        # Call the process directly with user input
        result = call_uipath_process(user_input)
        
        # Report the result
        if result.get("status") == "error":
            print(f"Error: {result.get('message', 'Unknown error')}")
        else:
            print("Process executed successfully!")
            print(f"Result: {result}")
        
        return result

    # Use the custom handler instead of initiate_chat
    chat_result = custom_chat_handler()
    return chat_result

if __name__ == "__main__":
    result = run_uipath_agent_with_human_input()
    print(result)

