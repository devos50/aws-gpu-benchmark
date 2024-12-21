import argparse
import sys
import time

import boto3

import paramiko


templates = {
    "p3": {
        "LaunchTemplateName": "p3_instance",
        "InstanceType": "p3.2xlarge",
    },
    "g4dn": {
        "LaunchTemplateName": "g5_instance",
        "InstanceType": "g4dn.2xlarge",
    },
    "g5": {
        "LaunchTemplateName": "g5_instance",
        "InstanceType": "g5.2xlarge",
    }
}

REGIONS = ["us-east-1", "us-west-2", "ap-northeast-1", "eu-central-1", "sa-east-1"]


def execute_ssh_command(instance_ip, key_file, username, command, blocking=True):
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
        if blocking:
            output = stdout.read().decode()
            error = stderr.read().decode()
            
            print(f"Command Output:\n{output}")
            if error:
                print(f"Command Error:\n{error}")
        
    except Exception as e:
        print(f"Failed to execute command on {instance_ip}: {e}")
    finally:
        ssh_client.close()


def get_all_ec2_instances():
    all_instances = []

    for region in REGIONS:
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
                        'AvailabilityZone': instance['Placement']['AvailabilityZone'],
                        'InstanceType': instance['InstanceType'],
                        'LaunchTime': instance['LaunchTime'].strftime('%Y-%m-%d %H:%M:%S'),
                        'PublicIP': instance.get('PublicIpAddress', 'N/A'),
                        'PrivateIP': instance.get('PrivateIpAddress', 'N/A'),
                        'Lifecycle': instance.get('InstanceLifecycle', 'On-Demand')
                    }
                    all_instances.append(instance_info)
        except Exception as e:
            print(f"Error fetching instances in region {region}: {e}")
    
    return all_instances


def spawn_spot_instance(region_name, template_name):
    template = templates[template_name]
    ec2 = boto3.client('ec2', region_name=region_name)
    
    # Configuration for Spot Instance
    spot_instance_config = {
        'LaunchTemplate': {
            'LaunchTemplateName': template['LaunchTemplateName'],
            'Version': '$Latest'  # Specify the version or '$Latest'
        },
        'InstanceType': template['InstanceType'],
        'InstanceMarketOptions': {
            'MarketType': 'spot',
            'SpotOptions': {
                'MaxPrice': '1.5',
                'SpotInstanceType': 'one-time',
                'InstanceInterruptionBehavior': 'terminate',
            }
        },
        'MinCount': 1,
        'MaxCount': 1,
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
    
def conduct_model_transfer_speed_experiment(intra_region=False):
    instances = get_all_ec2_instances()

    # Kill the transfer.py process on all instances
    for instance in instances:
        cmd = "pkill -f transfer.py"
        execute_ssh_command(instance["PublicIP"], '/Users/martijndevos/.ssh/Amazon.pem', 'ubuntu', cmd)

    time.sleep(5)
    print("Starting the transfer server again")
    for instance in instances:
        cmd = "source activate pytorch && cd /home/ubuntu/aws-gpu-benchmark && nohup python3 transfer.py server > server.log 2>&1 &"
        execute_ssh_command(instance["PublicIP"], '/Users/martijndevos/.ssh/Amazon.pem', 'ubuntu', cmd, blocking=False)

    time.sleep(5)

    print("Starting transfers...")
    for from_instance in instances:
        for to_instance in instances:
            if intra_region and from_instance["Region"] != to_instance["Region"]:
                continue
            if not intra_region and from_instance["Region"] == to_instance["Region"]:
                continue

            if from_instance["InstanceId"] == to_instance["InstanceId"]:
                continue

            print("Conducting experiment {} => {}".format(from_instance["Region"], to_instance["Region"]))
            cmd = "source activate pytorch && cd /home/ubuntu/aws-gpu-benchmark && python3 transfer.py client --ip {} --from-instance {} --to-instance {} --from-az {} --to-az {}".format(to_instance["PublicIP"], from_instance["InstanceType"], to_instance["InstanceType"], from_instance["AvailabilityZone"], to_instance["AvailabilityZone"])
            print("Running command: {}".format(cmd))
            execute_ssh_command(from_instance["PublicIP"], '/Users/martijndevos/.ssh/Amazon.pem', 'ubuntu', cmd)
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage AWS EC2 Spot Instances")
    parser.add_argument(
        "action",
        choices=["list", "spawn-instance", "test_model_transfer_speed", "terminate", "terminate_all", "cmd"],
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
    parser.add_argument(
        "--template",
        choices=["p3", "g4dn", "g5"],
        default="p3",
        help="The instance template."
    )
    parser.add_argument(
        "--cmd",
        required = False,
        help="The command to run on the instance."
    )
    parser.add_argument("--intra-region", action="store_true", help="Run the transfer speed test within the same region.")

    args = parser.parse_args()

    if args.action == "list":
        all_instances = get_all_ec2_instances()
        if all_instances:
            print(f"\nFound {len(all_instances)} instances:")
            for instance in all_instances:
                print(instance)
        else:
            print("No instances found.")
    elif args.action == "spawn-instance":
        spawn_spot_instance(args.region, args.template)
    elif args.action == "test_model_transfer_speed":
        conduct_model_transfer_speed_experiment(intra_region=args.intra_region)
    elif args.action == "terminate":
        terminate_instance(args.instance_id, region=args.region)
    elif args.action == "terminate_all":
        all_instances = get_all_ec2_instances()
        print(f"Terminating {len(all_instances)} instances...")
        for instance in all_instances:
            terminate_instance(instance['InstanceId'], region=instance['Region'])
    elif args.action == "cmd":
        execute_ssh_command(args.ip, '/Users/martijndevos/.ssh/Amazon.pem', 'ubuntu', args.cmd)
    else:
        print("Invalid argument. Use 'list' to list running instances or 'spawn-instance' to spawn a Spot instance.")
        sys.exit(1)
