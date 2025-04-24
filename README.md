# Capstone_C964

### WGU Computer Science Capstone

Navigate to `https://capstone-c964.onrender.com/` to use the Diabetes Risk Score Tool

## How to run locally

1. Clone the Repository
   To begin, open your terminal and run the command:

```bash
git clone https://github.com/mattyj7/capstone_c964.git
# Then navigate into the cloned directory:
cd capstone_c964
```

2. Open the Project in an IDE
   Use your preferred IDE, such as Visual Studio Code. To open the project in VS Code, run the following command:

```bash
code .
```

3. Install Required Python Packages
   Ensure Python 3 is installed on your machine.
   Install the required dependencies listed in the requirements.txt file by running:

```bash
pip install -r requirements.txt
```

4. Modify POST request to backend in `/templates/index.html`

```bash
# Change to http://localhost:8000/predict
# for local dev testing
# line 336
        fetch("https://capstone-c964.onrender.com/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(formData),
        })
```

5. Run the Application
   Once all packages are installed, you can start the application by running:

```bash
python3 app.py
```
