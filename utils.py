try:
    import requests
    import logging
    import json
    import openai
    import os
except:
    import requests
    import logging
    import json
    import openai
    import os


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

                    return final_result, total_tokens_used, cleaned_json, result_data

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


def delete_line_items(automation_job):
    result = post_backend(
        "azureoperations/delete-line-items/",
        {},
        {
            "automation_job_id": automation_job["id"],
        },
    )

    return result


def split_data(cleaned_json, max_pages=1):
    chunks = []
    current_chunk = {"analyzeResult": {"content": "", "tables": [], "documents": []}}
    current_page_count = 0

    for page_num, page_content in cleaned_json["analyzeResult"]["pages"].items():
        current_chunk["analyzeResult"]["content"] += page_content["content"] + " "
        current_page_count += 1

        for table in cleaned_json["analyzeResult"]["tables"]:
            if int(page_num) in table["page_numbers"]:
                current_chunk["analyzeResult"]["tables"].append(table)

        for document in cleaned_json.get("analyzeResult", {}).get("documents", []):
            for field_name, field_info in document.items():
                if field_info.get("pageNumber", 0) == int(page_num):
                    current_chunk["analyzeResult"]["documents"].append(
                        {field_name: field_info}
                    )

        if current_page_count >= max_pages:
            chunks.append(current_chunk)
            current_chunk = {
                "analyzeResult": {"content": "", "tables": [], "documents": []}
            }
            current_page_count = 0

    if (
        current_chunk["analyzeResult"]["content"]
        or current_chunk["analyzeResult"]["tables"]
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

    for field in automation_fields:
        if field["field_type"] == "list":
            combined_data[field["field_name"]] = []
        else:
            combined_data[field["field_name"]] = ""

    field_values = {
        field["field_name"]: []
        for field in automation_fields
        if field["field_type"] != "list"
    }

    for chunk in chunks:
        for key, values in chunk.items():
            if key in combined_data and key != "LineItems":
                if isinstance(combined_data[key], list):
                    combined_data[key].extend(values)
                else:
                    if values:
                        field_values[key].append(values)

    for key, values in field_values.items():
        if values:
            most_frequent_value = max(set(values), key=values.count)
            combined_data[key] = most_frequent_value

    for chunk in chunks:
        for key, values in chunk.items():
            if key == "LineItems":
                combined_data[key].extend(values)

    if "LineItems" in combined_data and isinstance(combined_data["LineItems"], list):
        combined_data["LineItems"] = [
            item
            for item in combined_data["LineItems"]
            if any(value.strip() for value in item.values())
        ]

    return combined_data


class ChatReadRetrieveReadApproach:
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MAX_TOKENS = 4096 - 256

    NO_RESPONSE = "0"

    def __init__(
        self,
        chatgpt_model="gpt-4-turbo",
        chatgpt_token_limit=MAX_TOKENS,
    ):
        self.chatgpt_model = chatgpt_model
        self.chatgpt_token_limit = chatgpt_token_limit

        openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"

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

        line_items_instruction = (
            "\n7. The features are columns for an array named 'LineItems'. "
            "These dictionary values must fill an array for the 'LineItems' key, which must be returned."
            if line_items
            else ""
        )

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
            """
        )

        fields_with_descriptions = [
            f"{field}: {description}"
            for field, description in zip(desired_fields, desired_fields_descriptions)
        ]

        fields_descriptions_joined = "\n\n".join(fields_with_descriptions)

        fields_content = (
            'Document content:\n\n"""'
            + json.dumps(document, indent=4)
            + '"""\n\n'
            + "Fields and Descriptions"
            + ':\n\n"""'
            + fields_descriptions_joined
        )

        messages = []
        messages.append(
            {
                "role": self.SYSTEM,
                "content": f"{system_message_extract}",
            }
        )
        messages.append({"role": self.USER, "content": fields_content})

        response_content = ""
        total_response = ""
        total_tokens_used = 0

        first_iteration = True

        while True:
            if first_iteration:
                chat_completion = openai.ChatCompletion.create(
                    engine=self.chatgpt_model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    max_tokens=self.chatgpt_token_limit,
                    temperature=0.0,
                )
                response_content = chat_completion.choices[0].message["content"].strip()
                response_content = extract_valid_json(response_content)
                first_iteration = False
            else:
                chat_completion = openai.ChatCompletion.create(
                    engine=self.chatgpt_model,
                    messages=messages,
                    max_tokens=self.chatgpt_token_limit,
                    temperature=0.0,
                )
                response_content = chat_completion.choices[0].message["content"].strip()

            # Token tracking
            tokens_used_output = num_tokens_from_response(
                response_content, model=self.chatgpt_model
            )
            total_tokens_used += tokens_used_output

            if chat_completion.choices[0].finish_reason == "length":
                total_response += remove_overlap(total_response, response_content)
                continuation_context = extract_continuation_context(response_content)

                continuation_prompt = """
                    You are a data extraction assistant. The continuation context provided is an earlier response you made while you were extracting data; however, your response was too long and you were unable to complete the extraction. You MUST continue the extraction. Here are your exact instructions:
                    
                    1. You will respond only as a string
                    2. You will include the last 10 characters from the continuation context.
                    """

                messages = [
                    {"role": self.SYSTEM, "content": system_message_extract + "\n\n" + continuation_prompt},
                    {
                        "role": self.USER,
                        "content": fields_content + "\n\n" + continuation_context,
                    },
                ]

            else:
                total_response += remove_overlap(total_response, response_content)
                break
        
        logging.info(f"Extracted data: {total_response}")
        return total_response, total_tokens_used

    def fix_tables(self, document, markdown_line_items):
        """
        Receive a JSON document and fix the tables in the document.

        Parameters:
        - document (str): The JSON document to fix tables in.
        - markdown_line_items (str): The markdown table of the well formed / fixed_rows

        Returns:
        - str: The fixed tables in a JSON format.
        - tokens_used (int): The number of tokens used to generate the response.
        """

        system_message_fix_tables = """
        You are an advanced data correction assistant. Your task is to reformat incorrectly structured markdown tables within a JSON document. The user will provide this document, along with a sample of fixed rows demonstrating the correct table format. The tables in the document have various formatting errors, such as merged cells, improperly combined rows, and incorrectly split cells.

        Here are your specific instructions:
        1. Receive and analyze the JSON document provided by the user. This document contains markdown tables with formatting errors.
        2. Examine the sample of fixed rows also provided by the user. Use these rows as a reference for the correct format of the tables.
        3. Identify and rectify any formatting errors in the tables from the JSON document. This includes unmerging cells, separating combined rows, and correcting split cells.
        4. Ensure that the corrected tables use only the headers from the user-provided sample of fixed rows.
        5. Generate a new JSON document. Your response must be exclusively in the following JSON format.
        {
            \"tables\": [
                {
                    \"page_numbers\": [1],
                    \"markdown_table\": \"| Fixture Type / Item No. | ...\"
                },
                ...
            ]
        }
        6. Obey user instructions.
        """

        user_message_content = (
            'Document content:\n\n"""'
            + json.dumps(document, indent=4)
            + '"""\n\n'
            + "Fixed Rows:\n\n"
            "" + markdown_line_items + '"""'
        )

        messages = []

        messages.append(
            {
                "role": self.SYSTEM,
                "content": f"{system_message_fix_tables}",
            }
        )

        messages.append({"role": self.USER, "content": user_message_content})

        response_content = ""
        total_response = ""
        total_tokens_used = 0

        first_iteration = True

        while True:
            if first_iteration:
                chat_completion = openai.ChatCompletion.create(
                    engine=self.chatgpt_model,
                    response_format={"type": "json_object"},
                    messages=messages,
                    max_tokens=self.chatgpt_token_limit,
                    temperature=0.0,
                )
                response_content = chat_completion.choices[0].message["content"].strip()
                response_content = extract_valid_json(response_content)
                first_iteration = False
            else:
                chat_completion = openai.ChatCompletion.create(
                    engine=self.chatgpt_model,
                    messages=messages,
                    max_tokens=self.chatgpt_token_limit,
                    temperature=0.0,
                )
                response_content = chat_completion.choices[0].message["content"].strip()

            # Token tracking
            tokens_used_output = num_tokens_from_response(
                response_content, model=self.chatgpt_model
            )
            total_tokens_used += tokens_used_output

            if chat_completion.choices[0].finish_reason == "length":
                total_response += remove_overlap(total_response, response_content)
                continuation_context = extract_continuation_context(response_content)

                continuation_prompt = """
                    You are a data extraction assistant. The continuation context provided is an earlier response you made while you were cleaning data; however, your response was too long and you were unable to complete the extraction. You MUST continue the cleaning. Here are your exact instructions:
                    
                    1. You will respond only as a string
                    2. You will include the last 10 characters from the continuation context.
                    """

                messages = [
                    {"role": self.SYSTEM, "content": system_message_fix_tables + "\n\n" + continuation_prompt},
                    {
                        "role": self.USER,
                        "content": user_message_content + "\n\n" + continuation_context,
                    },
                ]

            else:
                total_response += remove_overlap(total_response, response_content)
                break

        return total_response

def extract_continuation_context(
    response, initial_context_length=256, final_context_length=1024
):
    """
    Extracts portions of the response to use as context for the next request.
    It takes the first 'initial_context_length' characters and the last 'final_context_length' characters of the response.

    Args:
    - response (str): The response text from which to extract context.
    - initial_context_length (int): Number of characters to extract from the start of the response.
    - final_context_length (int): Number of characters to extract from the end of the response.

    Returns:
    - str: Extracted context from the response.
    """
    if len(response) <= initial_context_length + final_context_length:
        # If the response is short enough, return it in full
        return response

    start_context = response[:initial_context_length]
    end_context = response[-final_context_length:]
    omitted_section_notice = "\n[...]\n\n[Content Omitted for Brevity]\n\n[...]\n"

    return f"{start_context}{omitted_section_notice}{end_context}"


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


def num_tokens_from_response(response, model):
    """
    Calculates the number of tokens in the ChatGPT response.

    Args:
        response (str): The response text from ChatGPT.
        model (str): The name of the GPT model to use for tokenization.

    Returns:
        int: The number of tokens in the response.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(response))


def fix_table_helper(line_items, automation_job, automation_fields):
    """
    Helper function to fix the tables in the document.

    Args:
        line_items (list): The line items to be fixed.
        automation_job (AutomationJob): The automation job object.
        automation_fields (list): The automation fields object.

    Returns:
        final_result (dict): The final result containing the fixed tables.
        cleaned_json (dict): The cleaned JSON response from Azure Document Intelligence.
        automation_job (AutomationJob): The automation job object.

    """

    def create_markdown_table(line_items):
        headers = set()
        for item in line_items:
            for attr in item["attributes"]:
                headers.add(attr["key"])
        headers = sorted(list(headers))

        header_row = "| " + " | ".join(headers) + " |"

        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        markdown_table_content = header_row + "\n" + separator_row + "\n"

        for item in line_items:
            attributes = {attr["key"]: attr["value"] for attr in item["attributes"]}

            table_row = (
                "| " +
                " | ".join(
                    [attributes.get(header, "") for header in headers]
                ) +
                " |"
            )
            markdown_table_content += table_row + "\n"

        return markdown_table_content

    markdown_line_items = create_markdown_table(line_items)

    cleaned_json = automation_job["cleaned_data"]

    chunks = split_data(cleaned_json, max_pages=1)

    chat_instance = ChatReadRetrieveReadApproach()

    new_tables = []

    for chunk in chunks:
        result = json.loads(chat_instance.fix_tables(chunk, markdown_line_items))
        new_tables.extend(result["tables"])

    automation_job["cleaned_data"]["analyzeResult"]["tables"] = new_tables
    cleaned_json["analyzeResult"]["tables"] = new_tables

    chunks = split_data(cleaned_json)

    all_chunks_data = []
    total_tokens_used = 0
    for chunk in chunks:
        chunk_data, tokens_used = process_chunk(chunk, automation_fields)
        all_chunks_data.append(chunk_data)
        total_tokens_used += tokens_used

    final_result = combine_all_chunks(all_chunks_data, automation_fields)

    return final_result, cleaned_json, automation_job


def extract_valid_json(json_string):
    try:
        json.loads(json_string)
        return json_string
    except json.JSONDecodeError as e:
        malformed_index = e.pos
        return json_string[:malformed_index]

def remove_overlap(prev_response, new_response, overlap_size=100):
    # The overlap_size can be adjusted based on the expected size of the overlap
    overlap_length = min(len(prev_response), overlap_size)

    # logging.info("Prev Response:\n\n" + prev_response + "\n\n------" + "New Response:\n\n" + new_response + "\n\n------")

    for i in range(overlap_length, 0, -1):
        if prev_response[-i:] == new_response[:i]:
            return new_response[i:]  # Remove the overlapping part
    return new_response