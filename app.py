import os
import uuid
import json
from flask import Flask, render_template, request, redirect, session, jsonify
# from aws_s3 import generate_bdd_from_jira, generate_bdd_scenario, generate_test_data, upload_file_to_s3
from jira import get_issues, get_sprintid, get_boardid, get_issues_bug
from embedGenerate import handle_defect_detection_button_click

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get_bdd_jira_boardid", methods=['POST'])
def get_bdd_jira_boardid():
    try:
        jira_url = request.form.get('jira_url')
        email = request.form.get('email')
        password = request.form.get('password')
        board_ids = get_boardid(jira_url, email, password)
        return jsonify(board_ids=board_ids)

    except Exception as e:
        return jsonify(error=str(e))


@app.route("/get_bdd_jira_sprintid", methods=['POST'])
def get_bdd_jira_sprintid():
    try:
        jira_url = request.form.get('jira_url')
        email = request.form.get('email')
        password = request.form.get('password')
        board_id = request.form.get('board_id')
        # print(jira_url)
        # print(email)
        # print(password)
        # print(board_id)
        sprint_ids = get_sprintid(jira_url, email, password, board_id)
        return jsonify(sprint_ids=sprint_ids)

    except Exception as e:
        return jsonify(error=str(e))

@app.route("/get_bdd_jira_issue_bug", methods=['POST'])
def get_bdd_jira_issue_bug():
    try:
        jira_url = request.form.get('jira_url')
        email = request.form.get('email')
        password = request.form.get('password')
        board_id = request.form.get('board_id')
        sprint_id = request.form.get('sprint_id')
        # print(jira_url)
        # print(email)
        # print(password)
        # print("board_id="+board_id)
        # print("sprint_id="+sprint_id)
        issue_bugs = get_issues_bug(jira_url, email, password, board_id, sprint_id)
        # Print out all the issues fetched for verification
        # print("Fetched issues: ", issue_ids)
        return jsonify(issue_bugs=issue_bugs)

    except Exception as e:
        return jsonify(error=str(e))
    
@app.route("/generate_defect", methods=['POST'])
async def generate_defect_detection():
    try:
        # Check if "Select All" is chosen
        selected_issue = request.form.get('issue_id1')
        all_issues_json = request.form.get('all_issues')

        # Function to extract the part after the colon
        def extract_issue_description(issue):
            return issue.split(": ", 1)[1] if ": " in issue else issue

        # Process issues based on selection
        if selected_issue == 'select_all' and all_issues_json:
            issues = json.loads(all_issues_json)
            issues = [extract_issue_description(issue) for issue in issues]
            print("Processing all issues...", issues)
        else:
            issues = [extract_issue_description(selected_issue)]
            print("Processing single issue:", issues)

        # Get the latest file path for embedding
        upload_folder = "./upload"
        files = [os.path.join(upload_folder, f) for f in os.listdir(upload_folder)]
        
        if len(files) == 0:
            return render_template('index.html', status="No files found for processing")
        
        latest_file = max(files, key=os.path.getctime)
        print(f"Using file for processing: {latest_file}")

        # Process the defects with automatic embedding handling
        url = await handle_defect_detection_button_click(
            filepath=latest_file,
            issues=issues
        )

        if url is None:
            return render_template('index.html', status="Failed to generate test data")
        
        return render_template(
            'index.html', 
            status="Test data generated successfully", 
            response=url
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return render_template('index.html', status="Error processing issues")
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)