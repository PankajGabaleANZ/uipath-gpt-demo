#!/usr/bin/env python3

import autogen
import json
import os
from typing import Dict, List, Optional
import base64
from pathlib import Path
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
from utils import get_config
config, api_key, endpoint, apiversion, model = get_config()

# Configuration for AutoGen with Azure OpenAI
config_list = config

# Define the invoice fields schema
INVOICE_FIELDS_SCHEMA = {
    "in_ClientName": "string - Name of the client being invoiced",
    "in_InvoiceDate": "string - Date when invoice was created (MM/DD/YYYY format)",
    "in_InvoiceDueDate": "string - Date when payment is due (MM/DD/YYYY format)",
    "in_RateAmount": "float - Hourly or unit rate amount",
    "in_TotalHours": "int - Total number of hours worked",
    "in_LineItemDescription": "string - Description of services or items provided",
    "In_Services": "string - Type of services provided",
    "in_InvoiceNumber": "string - Unique invoice identifier",
    "in_Service": "boolean - Whether this is a service-based invoice",
    "in_PO_Number": "string - Purchase order number if applicable"
}

class InvoiceExtractor:
    def __init__(self):
        # Create the main extraction agent
        self.extractor_agent = autogen.AssistantAgent(
            name="invoice_extractor",
            llm_config={
                "config_list": config_list,
                "temperature": 0.1,
            },
            system_message=f"""You are an expert invoice data extraction agent. 

Your task is to analyze invoice documents and extract the following information in JSON format:

{json.dumps(INVOICE_FIELDS_SCHEMA, indent=2)}

EXTRACTION RULES:
1. Extract data exactly as it appears in the document
2. Use MM/DD/YYYY format for dates
3. Convert numeric values to appropriate types (float for rates, int for hours)
4. If a field is not found, set it to null
5. For boolean fields, determine based on document content
6. Be precise and accurate - double-check your extraction

Return ONLY a valid JSON object with the extracted data, no additional text."""
        )
        
        # Create a file handler agent that asks for PDF files
        self.file_handler_agent = autogen.AssistantAgent(
            name="file_handler",
            llm_config={
                "config_list": config_list,
                "temperature": 0,
            },
            system_message="""You are a helpful file handler agent. Your job is to:

1. Ask the user to provide a PDF invoice file for processing
2. Guide them through the process of specifying the file path
3. Confirm the file exists and is accessible
4. Provide clear instructions and helpful feedback

Always be polite and helpful. If a file is not found, ask them to verify the path and try again."""
        )
        
        # Create a validator agent to verify extractions
        self.validator_agent = autogen.AssistantAgent(
            name="data_validator",
            llm_config={
                "config_list": config_list,
                "temperature": 0,
            },
            system_message="""You are a data validation expert. Your task is to:

1. Review extracted invoice data for accuracy and completeness
2. Check that all required fields are present
3. Validate data types and formats (dates, numbers, strings)
4. Flag any inconsistencies or missing critical information
5. Suggest corrections if needed

Provide feedback in this format:
- VALID: Yes/No
- ISSUES: List any problems found
- SUGGESTIONS: Recommended corrections"""
        )
        
        # Create a user proxy that can interact with the user
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="ALWAYS",  # Changed to ALWAYS to allow user interaction
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").strip().lower().endswith(("terminate", "exit", "quit")),
            code_execution_config=False,
        )

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images for processing"""
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        image_paths = []
        
        # Convert each page to image
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save as temporary image
            temp_img_path = f"temp_page_{page_num + 1}.png"
            img.save(temp_img_path)
            image_paths.append(temp_img_path)
        
        doc.close()
        return image_paths

    def cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary image files"""
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {file_path}: {e}")

    def ask_for_pdf_file(self) -> str:
        """Interactive method to ask user for PDF file"""
        
        # Start conversation to ask for PDF file
        initial_message = """Hello! I'm here to help you extract information from invoice PDF files.

Please provide the full path to your PDF invoice file that you'd like me to process.

For example:
- /path/to/your/invoice.pdf
- C:\\Documents\\invoice.pdf
- ./invoices/invoice_001.pdf

What's the path to your PDF file?"""
        
        self.user_proxy.initiate_chat(
            self.file_handler_agent,
            message=initial_message,
            max_turns=10
        )
        
        # Get the file path from user input
        # This is a simplified approach - in a real implementation, 
        # you might want to parse the conversation history more carefully
        print("\nPlease enter the path to your PDF file:")
        pdf_path = input().strip().strip('"\'')
        
        return pdf_path

    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """Extract invoice data from a PDF file"""
        
        print(f"Processing PDF: {pdf_path}")
        
        # Convert PDF to images
        try:
            image_paths = self.pdf_to_images(pdf_path)
            print(f"Converted PDF to {len(image_paths)} images")
        except Exception as e:
            raise Exception(f"Error converting PDF to images: {e}")
        
        # Process each page (usually invoices are single page, but handle multiple)
        all_extracted_data = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing page {i + 1}...")
            
            try:
                # Extract from this page
                page_data = self.extract_from_image(image_path)
                if page_data:  # Only add non-empty extractions
                    all_extracted_data.append({
                        "page": i + 1,
                        "data": page_data
                    })
            except Exception as e:
                print(f"Error processing page {i + 1}: {e}")
        
        # Clean up temporary files
        self.cleanup_temp_files(image_paths)
        
        # Combine data from all pages (or return the best one)
        if len(all_extracted_data) == 0:
            return {}
        elif len(all_extracted_data) == 1:
            return all_extracted_data[0]["data"]
        else:
            # For multiple pages, return the one with the most complete data
            best_data = max(all_extracted_data, 
                          key=lambda x: len([v for v in x["data"].values() if v is not None]))
            print(f"Selected data from page {best_data['page']} as most complete")
            return best_data["data"]

    def extract_from_image(self, image_path: str) -> Dict:
        """Extract invoice data from an image file"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Encode image
        base64_image = self.encode_image(image_path)
        
        # Create message with image
        message = {
            "content": [
                {
                    "type": "text",
                    "text": "Please extract the invoice information from this image and return it in the specified JSON format."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
        
        # Start the extraction conversation
        self.user_proxy.initiate_chat(
            self.extractor_agent,
            message=message,
            max_turns=2
        )
        
        # Get the last message from extractor
        last_message = self.user_proxy.last_message()["content"]
        
        try:
            # Parse JSON response
            extracted_data = json.loads(last_message)
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {last_message}")
            return {}

    def extract_from_text(self, text_content: str) -> Dict:
        """Extract invoice data from text content"""
        
        message = f"""Please extract the invoice information from this text content and return it in the specified JSON format:

{text_content}"""
        
        # Start the extraction conversation
        self.user_proxy.initiate_chat(
            self.extractor_agent,
            message=message,
            max_turns=2
        )
        
        # Get the last message from extractor
        last_message = self.user_proxy.last_message()["content"]
        
        try:
            # Parse JSON response
            extracted_data = json.loads(last_message)
            return extracted_data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {last_message}")
            return {}

    def validate_extraction(self, extracted_data: Dict) -> Dict:
        """Validate the extracted data using the validator agent"""
        
        validation_message = f"""Please validate this extracted invoice data:

{json.dumps(extracted_data, indent=2)}

Check for accuracy, completeness, and proper formatting according to the schema."""
        
        # Start validation conversation
        self.user_proxy.initiate_chat(
            self.validator_agent,
            message=validation_message,
            max_turns=2
        )
        
        validation_result = self.user_proxy.last_message()["content"]
        
        return {
            "validation_report": validation_result,
            "data": extracted_data
        }

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API consumption"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_invoice_file(self, file_path: str = None, validate: bool = True) -> Dict:
        """Process a single invoice file and extract information"""
        
        # If no file path provided, ask the user
        if file_path is None:
            file_path = self.ask_for_pdf_file()
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing invoice: {file_path.name}")
        
        # Determine file type and extract accordingly
        if file_path.suffix.lower() == '.pdf':
            extracted_data = self.extract_from_pdf(str(file_path))
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            extracted_data = self.extract_from_image(str(file_path))
        elif file_path.suffix.lower() in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            extracted_data = self.extract_from_text(text_content)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Validate if requested
        if validate and extracted_data:
            result = self.validate_extraction(extracted_data)
            return result
        
        return {"data": extracted_data, "validation_report": "Validation skipped"}

    def interactive_extraction_session(self):
        """Run an interactive session where the agent asks for files and processes them"""
        
        print("🤖 Invoice Extraction Agent Starting...")
        print("=" * 60)
        
        while True:
            try:
                print("\n" + "=" * 60)
                print("🔍 Ready to process a new invoice!")
                print("=" * 60)
                
                # Ask for PDF file
                file_path = self.ask_for_pdf_file()
                
                if file_path.lower() in ['quit', 'exit', 'terminate']:
                    print("👋 Goodbye! Thanks for using the Invoice Extractor!")
                    break
                
                # Process the file
                result = self.process_invoice_file(file_path)
                
                # Display results
                print("\n" + "🎯" * 20)
                print("EXTRACTION RESULTS")
                print("🎯" * 20)
                print(json.dumps(result["data"], indent=2))
                
                print("\n" + "✅" * 20)
                print("VALIDATION REPORT")
                print("✅" * 20)
                print(result["validation_report"])
                
                # Ask if they want to process another file
                print("\n" + "=" * 60)
                print("Would you like to process another invoice? (yes/no/quit)")
                continue_choice = input().strip().lower()
                
                if continue_choice in ['no', 'quit', 'exit']:
                    print("👋 Goodbye! Thanks for using the Invoice Extractor!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing file: {e}")
                print("Let's try again with a different file...")
                continue

    def process_multiple_invoices(self, file_paths: List[str], output_file: str = None) -> List[Dict]:
        """Process multiple invoice files"""
        
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_invoice_file(file_path)
                result["source_file"] = file_path
                results.append(result)
                print(f"✅ Successfully processed: {file_path}")
            except Exception as e:
                error_result = {
                    "source_file": file_path,
                    "error": str(e),
                    "data": None,
                    "validation_report": f"Error: {str(e)}"
                }
                results.append(error_result)
                print(f"❌ Error processing {file_path}: {e}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {output_file}")
        
        return results


def main():
    """Main function with interactive PDF processing"""
    
    # Check Azure OpenAI environment variables
    required_env_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print("⚠️  Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set them with:")
        print("export AZURE_OPENAI_API_KEY='your-azure-api-key'")
        print("export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        return
    
    print("🚀 Azure OpenAI Invoice Extractor")
    print("=" * 50)
    print("This tool will help you extract structured data from PDF invoices.")
    print("Supported formats: PDF, PNG, JPG, TXT")
    print("=" * 50)
    
    # Initialize extractor
    try:
        extractor = InvoiceExtractor()
        
        # Choose mode
        print("\nSelect mode:")
        print("1. Interactive session (recommended)")
        print("2. Process specific file")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Run interactive session
            extractor.interactive_extraction_session()
            
        elif choice == "2":
            # Process specific file
            file_path = input("Enter the full path to your PDF invoice: ").strip().strip('"\'')
            
            if os.path.exists(file_path):
                result = extractor.process_invoice_file(file_path)
                
                print("\n" + "="*50)
                print("EXTRACTION RESULTS")
                print("="*50)
                print(json.dumps(result["data"], indent=2))
                
                print("\n" + "="*50)
                print("VALIDATION REPORT")
                print("="*50)
                print(result["validation_report"])
                
                # Save results
                output_file = f"extracted_data_{Path(file_path).stem}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Results saved to: {output_file}")
                
            else:
                print(f"❌ File not found: {file_path}")
        else:
            print("❌ Invalid choice. Please run the script again.")
            
    except Exception as e:
        print(f"❌ Error initializing extractor: {e}")
        print("Please check your Azure OpenAI configuration.")


if __name__ == "__main__":
    main()