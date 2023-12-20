try:
    import requests
    import logging
    import json
except:
    import requests
    import logging
    import json


def get_backend(url, headers, params):
    import os
    import requests

    headers["X-Azure-Token"] = os.environ["AUTH_KEY"]

    response = requests.get(
        os.environ["BACKEND_URL"] + url, headers=headers, params=params
    )
    response.raise_for_status()

    return response.json()


def post_backend(url, headers, data):
    import os
    import requests

    headers["X-Azure-Token"] = os.environ["AUTH_KEY"]

    response = requests.post(
        os.environ["BACKEND_URL"] + url, headers=headers, json=data
    )
    response.raise_for_status()

    try:
        return response.json()
    except ValueError:
        return {}


def pretty_print_json(json_data):
    import json

    print(json.dumps(json_data, indent=4))


def log_error(error):
    import traceback
    import os

    print(error)
    traceback.print_exc()


def upload_to_azure_blob(file, file_name, tenant_name, user):
    from azure.storage.blob import (
        BlobServiceClient,
        ContentSettings,
        BlobSasPermissions,
        generate_blob_sas,
        BlobType,
    )
    from datetime import datetime, timedelta
    import os

    try:
        connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        container_name = f"{tenant_name.lower()}-docs"
        blob_name = f"raw-files/{user['id']}-{file_name}"

        # Determine content type based on file extension
        content_type = "application/pdf" if file_name.endswith(".pdf") else "image/jpeg"

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        metadata = {
            "userId": str(user["id"]),
            "uploadedBy": user["email"],
            "userEmail": user["email"],
            "contentType": content_type.split("/")[1],
            "fileName": file_name,
        }

        blob_client.upload_blob(
            file,
            blob_type="BlockBlob",
            overwrite=True,
            metadata=metadata,
            content_settings=ContentSettings(
                content_type=content_type, content_disposition="inline"
            ),
        )

        account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
        account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=365),
            start=datetime.utcnow(),
            blob_type=BlobType.BlockBlob,
        )

        sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        print(
            f"File {blob_name} successfully uploaded to Azure Blob Storage with SAS URL: {sas_url}"
        )

        return {"url": sas_url, "name": blob_name}
    except Exception as e:
        log_error(e)
        raise


def extraction(file_url, automation_fields):
    """
    Extracts data from a given file URL using Azure Document Intelligence API and GPT4. Each file cannot have multiple invoices / claims / "items of interest".

    Args:
        file_url (str): The URL of the file to extract data from.
        automation_fields (list): A list of automation fields to extract from the document.

    Returns:
        list: A list of dictionaries containing the extracted data.
        total_tokens_used (int): The total number of tokens used to perform the extraction
        cleaned_json (dict): The cleaned JSON response from Azure Document Intelligence.

    Raises:
        Exception: If there is an error during the extraction process.

    """
    # !TODO - add support for large documents
    import os
    import requests
    import time

    try:
        key = os.environ["AZURE_MULTI_SERVICE_KEY"]
        endpoint = os.environ["AZURE_MULTI_SERVICE_ENDPOINT"]
        api_version = "2023-10-31-preview"

        # Define standard fields in the prebuilt-invoice
        standard_fields = {
            "CustomerName",
            "CustomerId",
            "PurchaseOrder",
            "InvoiceId",
            "InvoiceDate",
            "DueDate",
            "VendorName",
            "VendorTaxId",
            "VendorAddress",
            "VendorAddressRecipient",
            "CustomerAddress",
            "CustomerTaxId",
            "CustomerAddressRecipient",
            "BillingAddress",
            "BillingAddressRecipient",
            "ShippingAddress",
            "ShippingAddressRecipient",
            "PaymentTerm",
            "SubTotal",
            "TotalTax",
            "InvoiceTotal",
            "AmountDue",
            "ServiceAddress",
            "ServiceAddressRecipient",
            "RemittanceAddress",
            "RemittanceAddressRecipient",
            "ServiceStartDate",
            "ServiceEndDate",
            "PreviousUnpaidBalance",
            "CurrencyCode",
            "KVKNumber",
            "PaymentDetails",
            "TotalDiscount",
            "TaxItems",
        }

        query_fields = [
            field["field_name"]
            for field in automation_fields
            if field["field_name"] not in standard_fields
        ]
        query_fields_param = ",".join(query_fields)

        url = f"{endpoint}/documentintelligence/documentModels/prebuilt-invoice:analyze?api-version={api_version}&stringIndexType=utf16CodeUnit&queryFields={query_fields_param}&features=queryFields"
        headers = {
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/json",
        }
        data = {"urlSource": file_url}

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 202:
            operation_location = response.headers["Operation-Location"]

            # Polling the operation
            while True:
                result_response = requests.get(
                    operation_location,
                    headers={"Ocp-Apim-Subscription-Key": key},
                )

                if result_response.status_code == 200:
                    result_data = result_response.json()

                    if result_data.get("status") in ["running", "notStarted"]:
                        time.sleep(1)
                        continue

                    # Data cleaning
                    cleaned_json = clean_azure_response(result_data)

                    # Chunking by number of pages
                    chunks = split_data(cleaned_json)

                    # Process each chunk with GPT-4
                    all_chunks_data = []
                    total_tokens_used = 0
                    for chunk in chunks:
                        chunk_data, tokens_used = process_chunk(
                            chunk, automation_fields
                        )
                        all_chunks_data.append(chunk_data)
                        total_tokens_used += tokens_used

                    final_result = combine_all_chunks(
                        all_chunks_data, automation_fields
                    )

                    return final_result, total_tokens_used, cleaned_json

                elif result_response.status_code not in [200, 202]:
                    raise Exception(
                        f"HTTP Error during result fetching: {result_response.status_code}, {result_response.text}"
                    )

                time.sleep(1)
        else:
            raise Exception(
                f"Initial POST request HTTP Error: {response.status_code}, {response.text}"
            )

    except Exception as e:
        log_error(e)
        return False


def clean_azure_response(raw_response):
    """
    Cleans the raw response from Azure Document Intelligence and structures it into a more usable format.
    Reduces the dimensionality of the response to reduce request size to OpenAI.
    Enriches data with page number metadata.

    Args:
        raw_response (dict): The raw JSON response from Azure Document Intelligence.

    Returns:
        dict: A cleaned and structured version of the response.
    """

    def process_documents(documents):
        cleaned_documents = []
        for document in documents:
            fields = document.get("fields", {})
            for field_name, field_info in fields.items():
                content = field_info.get("content", "")
                page_number = field_info.get("boundingRegions", [{}])[0].get(
                    "pageNumber", None
                )
                cleaned_documents.append(
                    {field_name: {"content": content, "pageNumber": page_number}}
                )
        return cleaned_documents

    def process_tables(tables):
        """
        Process a list of tables and convert them to markdown format.

        Args:
            tables (list): A list of tables to be processed.

        Returns:
            list: A list of dictionaries containing the markdown table and the page numbers on which the table appears.
        """

        def create_markdown_table(table):
            """
            Convert table data to markdown format.
            """
            table_dict = {}
            for cell in table["cells"]:
                row_index = cell["rowIndex"]
                col_index = cell["columnIndex"]
                content = cell["content"]
                if row_index not in table_dict:
                    table_dict[row_index] = {}
                table_dict[row_index][col_index] = content

            markdown_table = ""
            max_col = max(max(table_dict[row].keys()) for row in table_dict) + 1
            for row in sorted(table_dict.keys()):
                row_data = (
                    "| "
                    + " | ".join(table_dict[row].get(col, "") for col in range(max_col))
                    + " |"
                )
                markdown_table += row_data + "\n"
                if row == 0:
                    header_separator = (
                        "| " + " | ".join("---" for _ in range(max_col)) + " |"
                    )
                    markdown_table += header_separator + "\n"

            return markdown_table

        def create_markdown_and_extract_pages(table):
            """
            Convert table data to markdown and extract the page numbers on which the table appears.
            """
            page_numbers = set()
            for cell in table["cells"]:
                for region in cell.get("boundingRegions", []):
                    page_numbers.add(region.get("pageNumber", 0))

            markdown_table = create_markdown_table(table)
            return {
                "markdown_table": markdown_table,
                "page_numbers": sorted(list(page_numbers)),
            }

        return [create_markdown_and_extract_pages(table) for table in tables]

    def process_pages(pages):
        page_contents = {}

        for page in pages:
            page_number = page.get("pageNumber", 0)
            content = " ".join([word["content"] for word in page.get("words", [])])
            page_contents[page_number] = {"pageNumber": page_number, "content": content}

        return page_contents

    tables = raw_response.get("analyzeResult", {}).get("tables", [])
    documents = raw_response.get("analyzeResult", {}).get("documents", [])
    pages = raw_response.get("analyzeResult", {}).get("pages", [])

    cleaned_tables = process_tables(tables)
    cleaned_documents = process_documents(documents)
    processed_pages = process_pages(pages)

    cleaned_json = {
        "analyzeResult": {
            "tables": cleaned_tables,
            "documents": cleaned_documents,
            "pages": processed_pages,
        }
    }

    return cleaned_json


def handle_line_items(extraction_result, automation_job, automation_field):
    """
    Handles extraction of list type fields (line items).

    Args:
        extraction_result (dict): The extraction result containing the extracted data.
        automation_job (AutomationJob): The automation job associated with the extraction.
        automation_field (AutomationField): The automation field representing the line items.

    Returns:
        None
    """
    line_items_data = extraction_result.get(automation_field["field_name"])

    ocr_result = post_backend(
        "azureoperations/create-ocr-result/",
        {},
        {
            "automation_job_id": automation_job["id"],
            "automation_field_id": automation_field["id"],
            "extracted_data": json.dumps(line_items_data),
        },
    )

    post_backend(
        "azureoperations/line-items-data/",
        {},
        {
            "ocr_result_id": ocr_result["id"],
            "line_items_data": json.dumps(line_items_data),
        },
    )


def handle_single_field(extraction_result, automation_job, automation_field):
    """
    Handles extraction of single value fields.

    Args:
        extraction_result (dict): The dictionary containing the extraction result.
        automation_job (AutomationJob): The automation job object.
        automation_field (AutomationField): The automation field object.

    Returns:
        None
    """
    field_value = extraction_result.get(automation_field["field_name"], "")
    post_backend(
        "azureoperations/create-ocr-result/",
        {},
        {
            "automation_job_id": automation_job["id"],
            "automation_field_id": automation_field["id"],
            "extracted_data": field_value,
        },
    )


def split_data(cleaned_json, max_pages=3):
    """
    Splits the data into chunks, each chunk containing data from up to max_pages.

    Args:
        cleaned_json (dict): The JSON data to be chunked.
        max_pages (int): The maximum number of pages for each chunk.

    Returns:
        list: A list of chunks, each chunk being a dictionary.
    """
    chunks = []
    current_chunk = {"analyzeResult": {"content": "", "tables": [], "documents": []}}
    current_page_count = 0

    # Iterate through pages and gather associated tables and documents
    for page_num, page_content in cleaned_json["analyzeResult"]["pages"].items():
        # Append page content to the chunk
        current_chunk["analyzeResult"]["content"] += page_content["content"] + " "

        # Process tables for this page
        for table in cleaned_json["analyzeResult"]["tables"]:
            if page_num in table["page_numbers"]:
                current_chunk["analyzeResult"]["tables"].append(table)

        # Process documents for this page
        for document in cleaned_json["analyzeResult"]["documents"]:
            for field_name, field_info in document.items():
                if field_info.get("pageNumber", 0) == page_num:
                    current_chunk["analyzeResult"]["documents"].append(
                        {field_name: field_info}
                    )

        # Check if max pages limit is reached and reset for a new chunk
        if current_page_count >= max_pages:
            chunks.append(current_chunk)
            current_chunk = {
                "analyzeResult": {"content": "", "tables": [], "documents": []}
            }
            current_page_count = 0

    # Add any remaining data to the last chunk
    if (
        current_chunk["analyzeResult"]["tables"]
        or current_chunk["analyzeResult"]["documents"]
    ):
        chunks.append(current_chunk)

    return chunks


def process_chunk(chunk, automation_fields):
    """
    Processes a chunk of document data.

    Args:
        chunk (dict): A chunk of the document.
        automation_fields (list): Fields to extract from the document.

    Returns:
        dict: Extracted data from the chunk.
        total_tokens_used (int): The number of tokens used to process the chunk.
    """
    import json

    chat_instance = ChatReadRetrieveReadApproach()
    desired_fields = []
    desired_fields_descriptions = []
    chunk_data = {}

    total_tokens_used = 0

    for automation_field in automation_fields:
        if (
            automation_field["field_type"] == "list"
            and "sub_fields" in automation_field
        ):
            # Process each subfield within the list
            sub_fields = [sf["field_name"] for sf in automation_field["sub_fields"]]
            sub_fields_descriptions = [
                sf["field_description"] for sf in automation_field["sub_fields"]
            ]

            result, tokens_used = chat_instance.extract(
                chunk,
                sub_fields,
                desired_fields_descriptions=sub_fields_descriptions,
                line_items=True,
            )
            result_data = json.loads(result)
            total_tokens_used += tokens_used

            if not result_data.get("LineItems"):
                empty_line_item = {sub_field: "" for sub_field in sub_fields}
                chunk_data[automation_field["field_name"]] = [empty_line_item]
            else:
                chunk_data[automation_field["field_name"]] = result_data.get(
                    "LineItems"
                )

        else:
            desired_fields.append(automation_field["field_name"])
            desired_fields_descriptions.append(automation_field["field_description"])

    # If there are non-list fields to extract, process them separately
    if desired_fields:
        result, tokens_used = chat_instance.extract(
            chunk,
            desired_fields,
            desired_fields_descriptions=desired_fields_descriptions,
            line_items=False,
        )
        result_data = json.loads(result)
        total_tokens_used += tokens_used

        for key, value in result_data.items():
            chunk_data[key] = value

    return chunk_data, total_tokens_used


def combine_all_chunks(chunks, automation_fields):
    combined_data = {}

    # Initialize fields with default values based on their type
    for field in automation_fields:
        if field["field_type"] == "list":
            combined_data[field["field_name"]] = []
        else:
            combined_data[field["field_name"]] = ""

    # Initialize a structure to hold non-list field values for each field
    field_values = {
        field["field_name"]: []
        for field in automation_fields
        if field["field_type"] != "list"
    }

    # Process each chunk
    for chunk in chunks:
        for key, values in chunk.items():
            if key in combined_data and key != "LineItems":
                if isinstance(combined_data[key], list):
                    # Extend the list of items (for list fields other than LineItems)
                    combined_data[key].extend(values)
                else:
                    # Collect values for non-list fields for later processing
                    if values:  # Ignore empty values
                        field_values[key].append(values)

    # Resolve contradictions in non-list fields
    for key, values in field_values.items():
        if values:  # Ensure there are non-empty values
            # Select the most frequent non-empty value
            most_frequent_value = max(set(values), key=values.count)
            combined_data[key] = most_frequent_value

    # Process LineItems separately to ensure it's last
    for chunk in chunks:
        for key, values in chunk.items():
            if key == "LineItems":
                combined_data[key].extend(values)

    # Filter out completely empty rows from LineItems
    if "LineItems" in combined_data and isinstance(combined_data["LineItems"], list):
        combined_data["LineItems"] = [
            item
            for item in combined_data["LineItems"]
            if any(value.strip() for value in item.values())
        ]

    return combined_data


class ChatReadRetrieveReadApproach:
    # Chat roles
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MAX_TOKENS = 4096 - 100

    NO_RESPONSE = "0"

    def __init__(
        self,
        chatgpt_model="gpt-4-turbo",
        chatgpt_token_limit=MAX_TOKENS,
    ):
        self.chatgpt_model = chatgpt_model
        self.chatgpt_token_limit = chatgpt_token_limit

    def get_search_query(self, chat_completion: dict[str, any], user_query: str):
        """
        Retrieves the search query from the chat completion response.

        Args:
            chat_completion (dict[str, any]): The chat completion response.
            user_query (str): The original user query.

        Returns:
            str: The search query extracted from the chat completion response, or the original user query if no search query is found.
        """
        import json

        response_message = chat_completion["choices"][0]["message"]
        if function_call := response_message.get("function_call"):
            if function_call["name"] == "search_sources":
                arg = json.loads(function_call["arguments"])
                search_query = arg.get("search_query", self.NO_RESPONSE)
                if search_query != self.NO_RESPONSE:
                    return search_query
        elif query_text := response_message.get("content"):
            if query_text.strip() != self.NO_RESPONSE:
                return query_text
        return user_query

    system_message_extract = """
    You are a sophisticated data extraction assistant specializing in processing JSON documents received from the Azure Document Intelligence API. Your task involves the following steps:

    1. Analyze the provided JSON document carefully. Tables of information are provided in markdown format.
    2. Identify and extract data corresponding to a predefined list of user-requested fields. These fields are specified along with their descriptions, which may include variations and specific formatting requirements.
    3. Respect the order of the fields as provided in the user request. This order is crucial for the correct organization of the extracted data.
    4. For each field, apply any specified formatting rules diligently. This may include date formats (e.g., MM/DD/YYYY), numerical representations (e.g., decimal places), and specific text patterns.
    5. If a particular field's data cannot be found in the JSON document, respond with an empty string for that field to maintain the structure of the output.
    6. Compile the extracted data and present it in a structured JSON format, adhering to the sequence of the requested fields.

    Your response must be exclusively in JSON format, ensuring it is precise, well-structured, and aligns with the user's extraction and formatting criteria. Focus on accuracy and clarity in the representation of the extracted data.
    """

    def extract(
        self, document, desired_fields, desired_fields_descriptions=[], line_items=False
    ):
        """
        Receive a JSON document and extract the information of the desired_fields and return in a JSON format.

        Parameters:
        - document (str): The JSON document to extract information from.
        - desired_fields (list): A list of fields to extract from the document.
        - desired_fields_descriptions (list, optional): A list of descriptions for the desired fields. Defaults to [].
        - line_items (bool, optional): Whether the desired fields are line items. Defaults to False.

        Returns:
        - str: The extracted information in a JSON format.
        - tokens_used (int): The number of tokens used to generate the response.
        """

        # Additional instruction for line items, added only if descriptions are provided
        line_items_instruction = (
            "\n7. The features are columns for an array named 'LineItems'. "
            "These dictionary values must fill an array for the 'LineItems' key, which must be returned."
            if line_items
            else ""
        )

        # Construct the system message extract with the conditional instruction
        system_message_extract = (
            """
            You are a sophisticated data extraction assistant specializing in processing JSON documents received from the Azure Document Intelligence API. You are also given a list of fields a user requests. Your task involves the following steps:

            1. You will examine the JSON document and return all requested fields.
            2. You will identify and extract data corresponding to the provided user-requested fields. These fields are specified along with their descriptions, which will include variations and specific formatting requirements.
            3. You will respect the order of the fields as provided in the user request. This order is crucial for the correct organization of the extracted data.
            4. You will apply any specified formatting rules diligently. This includes date formats (e.g., MM/DD/YYYY), numerical representations (e.g., decimal places), and specific text patterns.
            5. You will use empty strings for fields that cannot be extracted.
            6. You will compile the extracted data and present it in a structured JSON format, adhering to the sequence of the requested fields."""
            + line_items_instruction
            + """

            Your response must be exclusively in JSON format.
            """
        )

        import json
        import openai
        import os

        openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"

        messages = []

        messages.append(
            {
                "role": self.SYSTEM,
                "content": f"{system_message_extract}",
            }
        )

        fields_with_descriptions = [
            f"{field}: {description}"
            for field, description in zip(desired_fields, desired_fields_descriptions)
        ]

        # Adding an additional newline between each field and its description
        fields_descriptions_joined = "\n\n".join(fields_with_descriptions)

        fields_content = (
            'Document content:\n\n"""'
            + json.dumps(document, indent=4)
            + '"""\n\n'
            + "Fields and Descriptions"
            + ':\n\n"""'
            + fields_descriptions_joined  # Using the modified joined string
            + '"""\n\n'
            + self.system_message_extract
        )

        messages.append({"role": self.USER, "content": fields_content})

        chat_completion = openai.ChatCompletion.create(
            engine=self.chatgpt_model,
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=self.chatgpt_token_limit,
            temperature=0.0,
        )

        result = chat_completion.choices[0].message["content"].strip()

        tokens_used = num_tokens_from_messages(messages, model=self.chatgpt_model)
        tokens_used += self.chatgpt_token_limit

        return result, tokens_used


def num_tokens_from_messages(messages, model="gpt-4"):
    """
    Calculates the number of tokens required to generate a response given a list of messages.

    Args:
        messages (list): A list of messages, where each message is a dictionary containing the keys "name" and "content".
        model (str): The name of the GPT model to use for tokenization. Defaults to "gpt-3.5-turbo-0613".

    Returns:
        int: The total number of tokens required to generate a response.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
