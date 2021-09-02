### Install
Install dependences in the system with:

pip3 install -r requirements.txt

or, alternatively, use a virtual environment:

python3 -m venv env # Create virtual enviroment env
. env/bin/activate # Activate enviroment env
pip3 install -r requirements.txt # Install dependences on environment env

deactivate # Deactivate environment

### Execute
From the directory services:

-Launch the server on a terminal with:
python3 server_rest_api.py --debug

-Use the client example to make requests to the server. Por example:
python3 client_rest_api.py --img ../images/image1.jpg --output image1-out.jpg

