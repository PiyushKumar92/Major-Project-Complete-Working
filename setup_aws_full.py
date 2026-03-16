"""
AWS Full Setup Script
Interactive setup for complete AWS integration
"""

import getpass
from aws_setup import aws_setup

def main():
    print("🌟 Missing Person System - AWS Full Setup")
    print("=" * 50)
    print("This will configure complete AWS integration with:")
    print("✓ S3 Storage")
    print("✓ Lambda Functions") 
    print("✓ ECS Containers")
    print("✓ IAM Roles")
    print("✓ CloudWatch Monitoring")
    print("=" * 50)
    
    # Get AWS credentials
    print("\n🔐 AWS Credentials Required:")
    print("Get these from: https://console.aws.amazon.com/iam/home#/security_credentials")
    
    access_key = input("AWS Access Key ID: ").strip()
    if not access_key:
        print("❌ Access Key is required!")
        return
    
    secret_key = getpass.getpass("AWS Secret Access Key: ").strip()
    if not secret_key:
        print("❌ Secret Key is required!")
        return
    
    # Get region
    print("\n🌍 AWS Region Selection:")
    regions = [
        "us-east-1 (N. Virginia) - Default",
        "us-west-2 (Oregon)",
        "eu-west-1 (Ireland)",
        "ap-south-1 (Mumbai)",
        "ap-southeast-1 (Singapore)"
    ]
    
    for i, region in enumerate(regions, 1):
        print(f"{i}. {region}")
    
    try:
        choice = int(input("\nSelect region (1-5, default=1): ") or "1")
        region_map = {
            1: "us-east-1",
            2: "us-west-2", 
            3: "eu-west-1",
            4: "ap-south-1",
            5: "ap-southeast-1"
        }
        selected_region = region_map.get(choice, "us-east-1")
    except ValueError:
        selected_region = "us-east-1"
    
    print(f"\n🎯 Selected region: {selected_region}")
    
    # Confirm deployment
    print("\n⚠️  IMPORTANT:")
    print("This will create AWS resources that may incur charges.")
    print("Make sure you understand AWS pricing before proceeding.")
    
    confirm = input("\nProceed with deployment? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("❌ Deployment cancelled.")
        return
    
    # Deploy infrastructure
    print("\n🚀 Starting deployment...")
    result = aws_setup.deploy_full_infrastructure(access_key, secret_key, selected_region)
    
    if result['success']:
        print("\n🎉 SUCCESS! AWS infrastructure deployed successfully!")
        
        # Update cloud engine configuration
        update_cloud_config(result)
        
        print("\n🔧 Configuration updated automatically.")
        print("🧪 Run 'python test_aws_integration.py' to test everything.")
        
    else:
        print(f"\n❌ Deployment failed. Check the errors above.")
        print("💡 Common issues:")
        print("   - Invalid AWS credentials")
        print("   - Insufficient IAM permissions") 
        print("   - Region restrictions")
        print("   - Service limits exceeded")

def update_cloud_config(deployment_result):
    """Update cloud engine with AWS configuration"""
    try:
        # Read current cloud engine
        with open('cloud_engine.py', 'r') as f:
            content = f.read()
        
        # Update S3 bucket name
        if 's3_bucket' in deployment_result['setup_status']:
            bucket_name = deployment_result['setup_status']['s3_bucket']
            content = content.replace(
                'bucket_name="missing-person-data"',
                f'bucket_name="{bucket_name}"'
            )
        
        # Update region
        region = deployment_result['region']
        content = content.replace(
            "boto3.client('s3')",
            f"boto3.client('s3', region_name='{region}')"
        )
        content = content.replace(
            "boto3.client('ec2')",
            f"boto3.client('ec2', region_name='{region}')"
        )
        content = content.replace(
            "boto3.client('lambda')",
            f"boto3.client('lambda', region_name='{region}')"
        )
        content = content.replace(
            "boto3.client('cloudwatch')",
            f"boto3.client('cloudwatch', region_name='{region}')"
        )
        
        # Write updated content
        with open('cloud_engine.py', 'w') as f:
            f.write(content)
        
        print("✓ Cloud engine configuration updated")
        
    except Exception as e:
        print(f"⚠️ Could not update configuration automatically: {e}")
        print("Please update cloud_engine.py manually with your AWS settings.")

if __name__ == "__main__":
    main()