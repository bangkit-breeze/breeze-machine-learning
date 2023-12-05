## Setup and Installation

1. Create virtual env Python

```
python -m venv env
```

2. Activate virtual env python

```
./env/Scripts/activate
```

3. Install Python depedencies

```
pip install -r requirements.txt
```

## Run 

Run app in development environment

```
uvicorn app.main:app --reload
```
