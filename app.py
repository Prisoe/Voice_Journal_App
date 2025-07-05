from flask import Flask, request, render_template, redirect, url_for, flash
import boto3
import os
import  uuid

# Start Flask
app = Flask(__name__)
app.secret_key = 'secret_key' # set later

#AWS S3 Config
S3_Bucket = os.getenv('TRANSCRIBE_BUCKET_NAME', 'TRANSCRIBE_BUCKET_NAME')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_ACCESS_KEY')


# S3
s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No files uploaded')
        return  redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Generate a uuid
        filename = f"{uuid.uuid4()}-{file.filename}"
        try:
            # upload to S3
            s3_client.upload_fileobj(file, S3_Bucket, filename)
            flash(f'File uploaded successfully to S3 as {filename}')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'An error occured: {str(e)}')
            return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)