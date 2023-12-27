import azure.functions as func

try:
    import requests
    import logging
    from utils import (
        get_backend,
        post_backend,
        pretty_print_json,
        upload_to_azure_blob,
        extraction,
        handle_line_items,
        handle_single_field,
        delete_line_items,
        fix_table_helper,
    )
except:
    import requests
    import logging
    from utils import (
        get_backend,
        post_backend,
        pretty_print_json,
        upload_to_azure_blob,
        extraction,
        handle_line_items,
        handle_single_field,
    )

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# @app.function_name(name="outlookchecker")
# @app.schedule(schedule="0 */30 * * * *", arg_name="mytimer", run_on_startup=False)
# def http_trigger(mytimer: func.TimerRequest) -> None:
#     logging.info("Python HTTP trigger function processed a request.")
#     outlook_state = get_backend("users/get-outlook-token", {}, {})["outlook_state"]

#     emails_to_return = outlook_steps(
#         outlook_state["oauth_access_token"], outlook_state["last_email_indexed"]
#     )

#     print(emails_to_return)

#     # return func.HttpResponse(f"Hello. This HTTP triggered function executed successfully.",
#     #          status_code=200)


def outlook_steps(oauth_token, last_email_indexed):
    # Send a request to microsoft graph API, to get the last 10 emails
    # Place them in emails var

    headers = {
        "Authorization": f"Bearer {oauth_token}",
        "Content-Type": "application/json",
    }
    url = "https://graph.microsoft.com/v1.0/me/messages"
    params = {
        "$select": "subject,sender,receivedDateTime",
        "$orderby": "receivedDateTime DESC",
        "$top": "10",
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    emails = response.json().get("value", [])

    # If the last email indexed (received from the backend) is empty, then set it to the first email in the list, and don't run on the rest of the emails, because we're probably initializing the checking of the emails
    if last_email_indexed == "":
        last_email_indexed_id = emails[0]["id"]
        post_backend(
            "users/update-last-indexed/",
            {},
            {"last_email_indexed": last_email_indexed_id},
        )
        return

    emails_to_return = []

    last_email_indexed_to_return = None
    for email in emails:
        if email["id"] == last_email_indexed:
            # if email["id"] == "a":
            break
        else:
            email_to_append = {}
            pretty_print_json(email)
            # Keep track of which emails we've indexed so we can send it to the backend once we're done
            last_email_indexed_to_return = email["id"]

            # Parse the email contents
            email_to_append["id"] = email["id"]
            email_to_append["subject"] = email["subject"]
            email_to_append["sender"] = email["sender"]["emailAddress"]["address"]
            email_to_append["received_datetime"] = email["receivedDateTime"]

            # Request to get the body of the email
            url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}"
            params = {"$select": "body"}
            new_headers = {
                "Authorization": f"Bearer {oauth_token}",
                "Content-Type": "application/json",
                "Prefer": 'outlook.body-content-type="text"',  # Tells Microsoft Graph API to return body as plain text
            }
            response = requests.get(url, headers=new_headers, params=params)
            response.raise_for_status()
            pretty_print_json(response.json())

            email_body = response.json()["body"]["content"]
            email_to_append["body"] = email_body

            # Request to get all attachments of the email
            url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            attachments = response.json()["value"]

            # For each attachment, convert it to bytes and store in a dict temporarily
            attachments_to_append = []
            for attachment in attachments:
                attachment_to_append = {}
                attachment_to_append["id"] = attachment["id"]
                attachment_to_append["filename"] = attachment["name"]

                url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments/{attachment['id']}/$value"

                response = requests.get(url, headers=headers)
                response.raise_for_status()

                attachment_bytes = response.content
                attachment_to_append["bytes"] = " "  # attachment_bytes
                # with open(attachment_name, "wb") as f:
                #     f.write(attachment_bytes)

                attachments_to_append.append(attachment_to_append)

            email_to_append["attachments"] = attachments_to_append

            emails_to_return.append(email_to_append)

    if last_email_indexed_to_return != None:
        post_backend(
            "users/update-last-indexed/",
            {},
            {"last_email_indexed": last_email_indexed_to_return},
        )

    return emails_to_return


@app.function_name(name="QueueFunc")
@app.queue_trigger(
    arg_name="msg",
    queue_name="localqueue",
    connection="AZURE_STORAGE_CONNECTION_STRING_QUEUE",
)
def run_trigger(msg: func.QueueMessage) -> None:
    import json

    req = json.loads(msg.get_body().decode("utf-8"))

    uploaded_file_objects = json.loads(req["uploaded_file_objects"])
    automation_jobs = json.loads(req["automation_jobs"])
    automation_fields = json.loads(req["automation_fields"])

    for uploaded_file_object, automation_job in zip(
        uploaded_file_objects, automation_jobs
    ):
        file_path = uploaded_file_object["path"]

        try:
            result, tokens_used, cleaned_json, result_data = extraction(
                file_path, automation_fields
            )

            for automation_field in automation_fields:
                if automation_field["field_type"] == "list":
                    handle_line_items(result, automation_job, automation_field)
                else:
                    handle_single_field(result, automation_job, automation_field)

            post_backend(
                "azureoperations/update-automation-job/",
                {},
                {
                    "automation_job_id": automation_job["id"],
                    "status": "completed",
                    "tokens_used": tokens_used,
                    "cleaned_data": cleaned_json,
                    "result_data": result_data,
                },
            )
        except Exception as e:
            import traceback

            exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
            line_number = exc_traceback.tb_lineno

            if "429" in str(e):
                post_backend(
                    "azureoperations/update-automation-job/",
                    {},
                    {
                        "automation_job_id": automation_job["id"],
                        "status": "openai_ratelimit",
                        "error_message": f"{str(e)} (Line: {line_number})",
                    },
                )
            else:
                logging.info(f"Error: {str(e)} (Line: {line_number})")


@app.function_name(name="FixTableQueueFunc")
@app.queue_trigger(
    arg_name="msg",
    queue_name="fixtable-localqueue",
    connection="AZURE_STORAGE_CONNECTION_STRING_QUEUE",
)
def fix_table(msg: func.QueueMessage) -> None:
    try:
        import json

        req = json.loads(msg.get_body().decode("utf-8"))

        line_items = json.loads(req["line_items"])
        automation_job_id = req["automation_job_id"]
        automation_fields = json.loads(req["automation_fields"])

        automation_job = get_backend(
            "azureoperations/get-automation-job/",
            {},
            {"automation_job_id": automation_job_id},
        )

        final_result, cleaned_json, new_automation_job = fix_table_helper(
            line_items, automation_job, automation_fields
        )

        result = delete_line_items(new_automation_job)

        for automation_field in automation_fields:
            if automation_field["field_type"] == "list":
                handle_line_items(final_result, new_automation_job, automation_field)

        post_backend(
            "azureoperations/update-automation-job/",
            {},
            {
                "automation_job_id": automation_job["id"],
                "status": "completed",
                "cleaned_data": cleaned_json,
            },
        )

    except Exception as e:
        import traceback

        exc_type, exc_value, exc_traceback = traceback.sys.exc_info()
        line_number = exc_traceback.tb_lineno

        if "429" in str(e):
            post_backend(
                "azureoperations/update-automation-job/",
                {},
                {
                    "automation_job_id": automation_job["id"],
                    "status": "openai_ratelimit",
                    "error_message": f"{str(e)} (Line: {line_number})",
                },
            )
        else:
            logging.info(f"Error: {str(e)} (Line: {line_number})")
