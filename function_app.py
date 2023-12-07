import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    import logging
    logging.info('Python HTTP trigger function processed a request.')
    outlook_state = get_backend("users/get-outlook-token", {}, {})['outlook_state']
   
    emails_to_return = outlook_steps(outlook_state['oauth_access_token'], outlook_state['last_email_indexed'])

    print(emails_to_return)

    return func.HttpResponse(f"Hello. This HTTP triggered function executed successfully.",
             status_code=200)

def outlook_steps(oauth_token, last_email_indexed):
    # Send a request to microsoft graph API, to get the last 10 emails
    # Place them in emails var
    import requests
    print(oauth_token)
    # return

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
    if last_email_indexed == '':
        last_email_indexed_id = emails[0]["id"]
        post_backend('users/update-last-indexed/', {}, {'last_email_indexed': last_email_indexed_id})
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
            email_to_append['id'] = email["id"]
            email_to_append['subject'] = email["subject"]
            email_to_append['sender'] = email["sender"]["emailAddress"]["address"]
            email_to_append['received_datetime'] = email["receivedDateTime"]

            # Request to get the body of the email
            url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}"
            params = {"$select": "body"}
            new_headers = {
                "Authorization": f"Bearer {oauth_token}",
                "Content-Type": "application/json",
                "Prefer": 'outlook.body-content-type="text"', # Tells Microsoft Graph API to return body as plain text
            }
            response = requests.get(url, headers=new_headers, params=params)
            response.raise_for_status()
            pretty_print_json(response.json())

            email_body = response.json()["body"]["content"]
            email_to_append['body'] = email_body

            # Request to get all attachments of the email
            url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            attachments = response.json()["value"]

            # For each attachment, convert it to bytes and store in a dict temporarily
            attachments_to_append = []
            for attachment in attachments:
                attachment_to_append = {}
                attachment_to_append['id'] = attachment["id"]
                attachment_to_append['filename'] = attachment["name"]

                url = f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments/{attachment['id']}/$value"

                response = requests.get(url, headers=headers)
                response.raise_for_status()

                attachment_bytes = response.content
                attachment_to_append['bytes'] = ' ' # attachment_bytes
                # with open(attachment_name, "wb") as f:
                #     f.write(attachment_bytes)

                attachments_to_append.append(attachment_to_append)

            email_to_append['attachments'] = attachments_to_append

            emails_to_return.append(email_to_append)

    if last_email_indexed_to_return != None:
        post_backend('users/update-last-indexed/', {}, {'last_email_indexed': last_email_indexed_to_return})

    # Return the email data
    return emails_to_return

def get_backend(url, headers, params):
    import os
    import requests
    headers["X-Azure-Token"] = os.environ['AUTH_KEY']

    response = requests.get(os.environ['BACKEND_URL'] + url, headers=headers, params=params)
    response.raise_for_status()
    
    return response.json()
    
def post_backend(url, headers, data):
    import os
    import requests
    headers["X-Azure-Token"] = os.environ['AUTH_KEY']

    response = requests.post(os.environ['BACKEND_URL'] + url, headers=headers, data=data)
    response.raise_for_status()
    
    # check if the response has any json data, and if it doesnt, return an empty dict
    try:
        return response.json()
    except:
        return {}
    
def pretty_print_json(json_data):
    import json
    print(json.dumps(json_data, indent=4))
