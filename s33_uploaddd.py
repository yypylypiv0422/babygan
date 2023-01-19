import os.path

import boto3
import glob
def uploadding(filees):
    imagees=glob.glob(filees+'/*')
    url_list=[]
    for i in imagees:

        bucket_name = 'three3d'  # bucket_name
        file_name = i     # file path to upload
        namee=os.path.split(file_name)[1]
        object_key = namee        # fine_name_in_s3


        aws_id = 'AKIAYRFDMC3T2AXA6UXR'  # aws_access_key_id
        aws_secret = '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'  # aws_secret_access_key

        client = boto3.client('s3', aws_access_key_id=aws_id, aws_secret_access_key=aws_secret)
        # client.delete_object(Bucket=bucket_name, Key="object_key")
        client.upload_file(Filename=file_name, Bucket=bucket_name, Key=object_key)

        print("File uploaded successfully")

        response = client.generate_presigned_url('get_object',
                                                 Params={'Bucket': bucket_name,
                                                         'Key': object_key},
                                                 ExpiresIn=3600
                                                 )
        url_list.append(response)
        os.remove(file_name)


    return url_list
