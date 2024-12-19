import argparse
import sys

import boto3

import paramiko


def execute_ssh_command(instance_ip, key_file, username, command):
    """
    Connects to an EC2 instance via SSH and executes a command.
    """
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to the instance
        ssh_client.connect(
            hostname=instance_ip,
            username=username,
            key_filename=key_file,
        )
        
        # Execute the command
        stdin, stdout, stderr = ssh_client.exec_command(command)
        
        # Get the output and error
        output = stdout.read().decode()
        error = stderr.read().decode()
        
        print(f"Command Output:\n{output}")
        if error:
            print(f"Command Error:\n{error}")
        
    except Exception as e:
        print(f"Failed to execute command on {instance_ip}: {e}")
    finally:
        ssh_client.close()


def list_all_ec2_instances():
    regions = ["us-east-1", "eu-central-1", "ap-northeast-1"]
    
    all_instances = []

    for region in regions:
        print(f"Checking region: {region}")
        ec2 = boto3.client('ec2', region_name=region)
        
        # Describe instances
        try:
            response = ec2.describe_instances()
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    if instance['State']['Name'] != 'running':
                        continue

                    instance_info = {
                        'InstanceId': instance['InstanceId'],
                        'State': instance['State']['Name'],
                        'Region': region,
                        'InstanceType': instance['InstanceType'],
                        'LaunchTime': instance['LaunchTime'].strftime('%Y-%m-%d %H:%M:%S'),
                        'PublicIP': instance.get('PublicIpAddress', 'N/A'),
                        'PrivateIP': instance.get('PrivateIpAddress', 'N/A'),
                    }
                    all_instances.append(instance_info)
        except Exception as e:
            print(f"Error fetching instances in region {region}: {e}")

    # Display the instances
    if all_instances:
        print(f"\nFound {len(all_instances)} instances:")
        for instance in all_instances:
            print(instance)
    else:
        print("No instances found.")
    
    return all_instances


def spawn_spot_instance(region_name):
    ec2 = boto3.client('ec2', region_name=region_name)
    
    # Configuration for Spot Instance
    spot_instance_config = {
        'LaunchTemplate': {
            'LaunchTemplateName': 'p3_instance',
            'Version': '$Latest'  # Specify the version or '$Latest'
        },
        'InstanceType': 'p3.2xlarge',  # Replace with the desired instance type
        'InstanceMarketOptions': {
            'MarketType': 'spot',
            'SpotOptions': {
                'MaxPrice': '1.5',
                'SpotInstanceType': 'one-time',
                'InstanceInterruptionBehavior': 'terminate',
            }
        },
        'MinCount': 1,  # Minimum number of instances
        'MaxCount': 1,  # Maximum number of instances
    }
    
    # Launch Spot Instances
    try:
        response = ec2.run_instances(**spot_instance_config)
        for instance in response['Instances']:
            print(f"Launched Spot Instance: {instance['InstanceId']} in state {instance['State']['Name']}")
    except Exception as e:
        print(f"Error launching spot instances: {e}")


def terminate_instance(instance_id, region="us-east-1"):
    """
    Terminates a specific EC2 instance (works for Spot and On-Demand instances).
    
    Args:
        instance_id (str): The ID of the EC2 instance to terminate.
        region (str): The AWS region where the instance is located (default: "us-east-1").
    
    Returns:
        str: The current state of the instance after termination.
    """
    ec2 = boto3.client('ec2', region_name=region)
    
    try:
        response = ec2.terminate_instances(InstanceIds=[instance_id])
        current_state = response['TerminatingInstances'][0]['CurrentState']['Name']
        print(f"Instance {instance_id} is now {current_state}.")
        return current_state
    except Exception as e:
        print(f"Error terminating instance {instance_id}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage AWS EC2 Spot Instances")
    parser.add_argument(
        "action",
        choices=["list", "spawn-instance", "cmd", "terminate"],
        help="Action to perform."
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region to target (default: us-east-1)."
    )
    parser.add_argument(
        "--instance-id",
        default="xxxxx",
        help="The instance ID."
    )
    parser.add_argument(
        "--ip",
        default="xxxxx",
        help="The instance IP."
    )

    args = parser.parse_args()

    if args.action == "list":
        list_all_ec2_instances()
    elif args.action == "spawn-instance":
        spawn_spot_instance(args.region)
    elif args.action == "cmd":
        execute_ssh_command(args.ip, '/Users/mdevos/.ssh/Amazon.pem', 'ubuntu', 'nvidia-smi')
    elif args.action == "terminate":
        terminate_instance(args.instance_id, region=args.region)
    else:
        print("Invalid argument. Use 'list' to list running instances or 'spawn-instance' to spawn a Spot instance.")
        sys.exit(1)
