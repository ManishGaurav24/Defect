import os
import uuid
import json
from flask import Flask, render_template, request, redirect, session, jsonify
# from aws_s3 import generate_bdd_from_jira, generate_bdd_scenario, generate_test_data, upload_file_to_s3
from jira import get_issues, get_sprintid, get_boardid, get_issues_bug
from embedGenerate import handle_start_embedding_button_click, handle_defect_detection_button_click

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
def generate_defect_detection():
    try:
        # Check if "Select All" is chosen
        selected_issue = request.form.get('issue_id1')
        all_issues_json = request.form.get('all_issues')  # This will be present if "Select All" is chosen

        # Function to extract the part after the colon
        def extract_issue_description(issue):
            return issue.split(": ", 1)[1] if ": " in issue else issue

        # If "Select All" is selected, process all issues
        if selected_issue == 'select_all' and all_issues_json:
            issues = json.loads(all_issues_json)
            # Extract only the issue descriptions (after the colon)
            issues = [extract_issue_description(issue) for issue in issues]
            print("Processing all issues...", issues)
        else:
            # Process individual issue
            issues = [extract_issue_description(selected_issue)]
            print(issues)
        
        # Pass the cleaned issues to the function
        url = handle_defect_detection_button_click(issue=issues)
        if url is None:
            return render_template('index.html', status="Failed to generate test data")
        return render_template('index.html', status="Test data generated successfully", response=url)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")
        return render_template('index.html', status="Error processing issues")
    
@app.route("/trigger_embedding", methods=["POST"])
def triggerEmbedding():
    upload_folder = "./upload"
    try:
        # Get all files in the upload directory
        files = [os.path.join(upload_folder, f) for f in os.listdir(upload_folder)]
        
        if len(files) == 0:
            return render_template('index.html', status="No files found for embedding")
        
        # Get the file with the latest creation time
        latest_file = max(files, key=os.path.getctime)
        
        # Call the embedding function with the latest file
        print((f"Embedding process started for file: {latest_file}"))
        url = handle_start_embedding_button_click(latest_file)
        
        return render_template('index.html',status="Embedding completed successfully")
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', status="Error while processing the file")
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)