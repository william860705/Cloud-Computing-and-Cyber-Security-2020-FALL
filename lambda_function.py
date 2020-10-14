# import the json utility package since we will be working with a JSON object
import json
# import the AWS SDK (for Python the package name is boto3)
import boto3
# import two packages to help us with dates and date formatting
from time import gmtime, strftime

# create a DynamoDB object using the AWS SDK
dynamodb = boto3.resource('dynamodb')
# use the DynamoDB object to select our table
table = dynamodb.Table('HelloWorldDatabase')
# store the current time in a human readable format in a variable
now = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
# from botocore.vendored import requests
import requests



# define the handler function that the Lambda service will use as an entry pointxw
def lambda_handler(event, context):
# extract values from the event object we got from the Lambda service and store in a variable

    city = event['city']
    r = requests.get('http://api.openweathermap.org/data/2.5/weather?q=%s&APPID=6105fcdc3b6a3c9f57f4994f380af9dd'%(city))
    r = r.json()
    
    data = {"weatherType":r['weather'][0]["main"], "temp":round(r['main']['temp']-273, 2), "icon":r['weather'][0]["icon"]}

# write name and time to the DynamoDB table using the object we instantiated and save response in a variable
    response = table.put_item(
        Item={
            'ID':r['weather'][0]["main"],
            'LatestGreetingTime':now
            })
# return a properly formatted JSON object
    return {
        'statusCode': 200,
        'weather': json.dumps(data)
    }