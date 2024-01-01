## Installation and SetUp of Website

### Clone the Repository
```shell
cd Downloads
git clone https://github.com/Avitra2002/Albania-s-food-security-prediction.git
```

### Create and Activate a Virtual Environment
Unix/MacOS:
```shell
$ cd ~/Downloads/Albania-s-food-security-prediction/webapp/web
```

First make sure that you have installed `pipenv` package.

```shell
pip install --user pipenv
```

```shell
export PATH='/voc/work/.local/bin':$PATH
```

From the root folder, install the packages specified in the `requirements.txt`.
```shell
pipenv install
```
Or
```shell
pip install -r requirements.txt
```
To activate the virtualenv, run
```shell
pipenv shell
```

To exit the virtual environment at the end of this mini project, simply type:_
```shell
exit
```

### Environment Configurations
In the "__init__.py" file, replace "your_secret_key_here" with your own key.

```shell
SECRET_KEY=your_secret_key_here
```

### Running the App
Go to the root folder and type:
```shell
flask run
```

You should see that some output will be thrown out, which one of them would be:

```shell
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

Now you can open your browser at `http://127.0.0.1:5000/` to see the web app.


