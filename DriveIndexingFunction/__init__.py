import logging
import os
import requests
import azure.functions as func

from .utils import pretty_print_json, log_error

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
)
import os


def main(req: func.HttpRequest) -> func.HttpResponse:
    index_google_folder(req)

    return func.HttpResponse(
        "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
        status_code=200,
    )


def index_google_folder(req):
    merge_token = req.params.get("merge_account_token")

    headers = {
        "Authorization": f"Bearer {os.environ['MERGE_API_KEY']}",
        "X-Account-Token": merge_token,
    }

    folder_id = req.params.get("folder_id")

    folders_url = (
        f"https://api.merge.dev/api/filestorage/v1/folders?remote_id={folder_id}"
    )
    folder_result = requests.get(
        folders_url,
        headers=headers,
    ).json()
    results = folder_result.get("results")
    merge_id = results[0].get("id")

    folder_indexing_url = (
        f"https://api.merge.dev/api/filestorage/v1/files?folder_id={merge_id}"
    )
    folder_indexing_result = requests.get(
        folder_indexing_url,
        headers=headers,
    ).json()
    file_results = folder_indexing_result.get("results")

    for result in file_results:
        file_id = result.get("id")
        mime_type = result.get("mime_type")

        file_extension = ".pdf"

        if mime_type == "application/vnd.google-apps.document":
            file_extension = ".pdf"

        download_url = (
            f"https://api.merge.dev/api/filestorage/v1/files/{file_id}/download"
        )

        tenant_name = req.params.get("tenant_name")
        user_id = req.params.get("user_id")
        user_email = req.params.get("user_email")

        upload_to_azure_blob_from_url_with_auth(
            download_url,
            headers,
            file_id + file_extension,
            tenant_name,
            user_id,
            user_email,
        )
        logging.info(f"Uploaded {file_id + file_extension} to Azure Blob Storage")

    # return Response(
    #     status=status.HTTP_200_OK,
    #     data=MergeLinkFolderSerializer(merge_link_folder).data,
    # )


def upload_to_azure_blob_from_url_with_auth(
    download_url, headers, file_name, tenant_name, user_id, user_email
):
    try:
        connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        container_name = f"{tenant_name.lower()}-docs"

        blob_name = f"raw-files/{user_id}-{file_name}"

        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        metadata = {
            "userId": str(user_id),
            "uploadedBy": user_email,
            "userEmail": user_email,
            "contentType": "pdf",
            "fileName": file_name,
        }

        response = requests.get(download_url, headers=headers)
        if response.status_code == 200:
            file_content = response.content
            blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)

            print(f"File {blob_name} successfully uploaded to Azure Blob Storage")
            return {
                "name": blob_name,
            }
        else:
            print(f"Failed to fetch the file from URL: {download_url}")
            return None
    except Exception as e:
        log_error(e)
        raise
