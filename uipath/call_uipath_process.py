import requests
import json
from typing import Optional, Dict, Any
import urllib
import time
from utils import get_uipath_config


# === Configuration ===
cfg = get_uipath_config()

client_id = cfg["client_id"]
user_key = cfg["client_secret"]
account_logical_name = cfg["account_logical_name"]
tenant_logical_name = cfg["tenant_logical_name"]
base_url = cfg["base_url"]
folder_id = cfg["folder_id"]

auth_url = f'{base_url}/identity_/connect/token'
orchestrator_base = f'{base_url}/{account_logical_name}/{tenant_logical_name}/'

#define input arguments for the job:
inputs = {
        "in_Emailto": "dhenn.mark.t.espiritu@gds.ey.com",
        "in_Subject": "Testing with uiPath"
}

def get_access_token() -> str:
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': user_key,
        'scope': 'OR.Folders OR.Robots OR.Machines OR.Execution OR.Assets OR.Jobs OR.Queues'
    }
    response = requests.post(auth_url, headers=headers, data=data)
    response.raise_for_status()
    return response.json()['access_token']

def get_uipath_folder_id(base_url, access_token):
    
    # Encode folder name to avoid issues with spaces/special chars
  
    folders_url = f"{base_url}/odata/Folders"
    headers = {"Authorization": f"Bearer {access_token}"}

    folder_resp = requests.get(folders_url, headers=headers)
    folder_resp.raise_for_status()
    results = folder_resp.json()["value"]

    return results[0]["Id"] if results else None

def get_release_key(access_token: str, process_key: str) -> str:
    url = f"{orchestrator_base}odata/Releases?$filter=ProcessKey eq '{process_key}'"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'X-UIPATH-OrganizationUnitId': folder_id
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    results = response.json().get('value', [])
    if not results:
        raise Exception(f"No process found with name: {process_key}")
    return results[0]['Key']

def start_uipath_job(access_token: str, process_release_key: str, input_args: Dict[str,Any]) -> Dict:   
    url = f"{orchestrator_base}odata/Jobs/UiPath.Server.Configuration.OData.StartJobs"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-UIPATH-OrganizationUnitId': folder_id,
        'X-UIPATH-TenantName': tenant_logical_name
    }
    payload = {
        "startInfo": {
            "ReleaseKey": process_release_key,
            "Strategy": "ModernJobsCount",
            "JobsCount": 1,
            "RobotIds": [],
            "InputArguments": json.dumps(input_args),
            "FolderId": int(folder_id)
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()

    # Extract and print job ID
    if 'value' in result and len(result['value']) > 0:
        job_id = result['value'][0]['Id']
        print(f"Job started successfully. Job ID: {job_id}")
    else:
        print("Job started but no job ID found in response")
    
    return result

def get_job_status_and_output(access_token: str, job_id: str, poll_interval: int = 5, timeout: int = 300) -> Optional[Dict]:
    """Get job outputs, polling until the job is complete or faulted."""

    url = f"{orchestrator_base}odata/Jobs({job_id})"
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
        'X-UIPATH-OrganizationUnitId': folder_id,
        'X-UIPATH-TenantName': tenant_logical_name
    }
    
    start_time = time.time()
    
    # Define terminal states (states where polling should stop)
    terminal_states = ['Successful', 'Faulted', 'Failed', 'Stopped', 'Suspended', 'Canceled']
    success_states = ['Successful']
    failure_states = ['Faulted', 'Failed', 'Stopped', 'Suspended', 'Canceled']
    
    while True:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            job_data = response.json()
            
            job_status = job_data.get('State', 'Unknown')
            print(f"Job {job_id} status: {job_status}")
            
            # Check if job has reached a terminal state
            if job_status in terminal_states:
                print(f"Job reached terminal state: {job_status}")
                
                # Handle successful completion
                if job_status in success_states:
                    print("Job completed successfully")
                    
                    # Parse output arguments if they exist
                    output_args = job_data.get('OutputArguments')
                    if output_args:
                        try:
                            parsed_outputs = json.loads(output_args)
                            job_data['ParsedOutputArguments'] = parsed_outputs
                            print("Job outputs retrieved successfully")
                        except json.JSONDecodeError:
                            print("Warning: Could not parse output arguments as JSON")
                
                # Handle failure states (including Faulted)
                elif job_status in failure_states:
                    print(f"Job failed with status: {job_status}")
                    
                    # Extract error information
                    error_info = {
                        'status': job_status,
                        'error_message': job_data.get('Info', 'No error details available'),
                        'creation_time': job_data.get('CreationTime'),
                        'start_time': job_data.get('StartTime'),
                        'end_time': job_data.get('EndTime'),
                        'host_machine_name': job_data.get('HostMachineName'),
                        'robot_name': job_data.get('Robot', {}).get('Name') if job_data.get('Robot') else None
                    }
                    
                    # Add error details to job_data for caller to handle
                    job_data['ErrorDetails'] = error_info
                    
                    # Log specific error information based on status
                    if job_status == 'Faulted':
                        print(f"Job faulted. Error: {error_info['error_message']}")
                    elif job_status == 'Stopped':
                        print("Job was manually stopped")
                    elif job_status == 'Suspended':
                        print("Job was suspended")
                    elif job_status == 'Canceled':
                        print("Job was canceled")
                
                # Return job data for both success and failure cases
                job_data['IsSuccess'] = job_status in success_states
                return job_data
            
            # Job is still running (Pending, Running, Resuming, etc.)
            elif job_status in ['Pending', 'Running', 'Resuming']:
                print(f"Job is {job_status.lower()}, continuing to poll...")
            else:
                print(f"Unknown job status: {job_status}, continuing to poll...")
            
            # Check for timeout
            if time.time() - start_time > timeout:
                print(f"Timeout reached ({timeout}s). Job status: {job_status}")
                # Return current job data with timeout flag
                job_data['IsTimeout'] = True
                job_data['IsSuccess'] = False
                return job_data
            
            # Wait before next poll
            time.sleep(poll_interval)
            
        except requests.exceptions.RequestException as e:
            print(f"Error checking job status: {e}")
            
            # Check if we should give up due to timeout
            if time.time() - start_time > timeout:
                print("Timeout reached during error condition")
                return None
                
            # Wait before retrying
            time.sleep(poll_interval)
            
        except KeyboardInterrupt:
            print("Polling interrupted by user")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


def start_job_and_wait_for_completion(access_token: str, process_release_key: str,
                                      input_args: Dict[str, Any], poll_interval: int = 5,
                                      timeout: int = 300) -> Optional[Dict]:
    """Convenience function to start a job and wait for its completion."""
    
    # Start the job
    start_result = start_uipath_job(access_token, process_release_key, input_args)
    
    # Extract job ID
    if 'value' in start_result and len(start_result['value']) > 0:
        job_id = start_result['value'][0]['Id']
        print(f"Started job with ID: {job_id}")
        
        # Wait for completion
        result = get_job_status_and_output(access_token, job_id, poll_interval, timeout)
        
        if result is None:
            print("Failed to get job result")
            return None
        
        # Check if job was successful
        if result.get('IsSuccess', False):
            print("Job completed successfully!")
            return result.get('ParsedOutputArguments', {})
        else:
            print("Job failed!")
            if 'ErrorDetails' in result:
                error_details = result['ErrorDetails']
                print(f"Error Status: {error_details['status']}")
                print(f"Error Message: {error_details['error_message']}")
            elif result.get('IsTimeout', False):
                print("Job timed out")
            
            return result  # Return full result so caller can handle the error
    else:
        print("Could not extract job ID from start response")
        return None

# === Function for Autogen Tool ===
def call_uipath_process(process_name: str, input_args: Optional[Dict[str,Any]] = None) -> Dict:
    
    #Call a UiPath process by name using Orchestrator API.

    try:
        token = get_access_token()
        release_key = get_release_key(token, process_name)
        result = start_job_and_wait_for_completion(token, release_key,input_args)
        return {
            "status": "success",
            "message": f"Started process '{process_name}'",
            "orchestrator_response": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
