"""
Quick AWS Configuration Script
Configure AWS credentials for immediate use
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

def configure_aws_credentials():
    """Configure AWS credentials interactively"""
    print("AWS Credentials Configuration")
    print("=" * 35)
    
    # Get credentials from user
    access_key = input("Enter AWS Access Key ID: ").strip()
    secret_key = input("Enter AWS Secret Access Key: ").strip()
    region = input("Enter AWS Region (default: us-east-1): ").strip() or 'us-east-1'
    
    if not access_key or not secret_key:
        print("❌ Both Access Key and Secret Key are required!")
        return False
    
    # Set environment variables
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    os.environ['AWS_DEFAULT_REGION'] = region
    
    # Test credentials
    try:
        sts = boto3.client('sts', region_name=region)
        identity = sts.get_caller_identity()
        
        print("✅ AWS credentials configured successfully!")
        print(f"Account ID: {identity['Account']}")
        print(f"User ARN: {identity['Arn']}")
        print(f"Region: {region}")
        
        # Save to AWS credentials file
        save_to_aws_config(access_key, secret_key, region)
        
        return True
        
    except Exception as e:
        print(f"❌ AWS credential test failed: {e}")
        return False

def save_to_aws_config(access_key, secret_key, region):
    """Save credentials to AWS config file"""
    try:
        aws_dir = os.path.expanduser('~/.aws')
        os.makedirs(aws_dir, exist_ok=True)
        
        # Write credentials file
        credentials_file = os.path.join(aws_dir, 'credentials')
        with open(credentials_file, 'w') as f:
            f.write('[default]\n')
            f.write(f'aws_access_key_id = {access_key}\n')
            f.write(f'aws_secret_access_key = {secret_key}\n')
        
        # Write config file
        config_file = os.path.join(aws_dir, 'config')
        with open(config_file, 'w') as f:
            f.write('[default]\n')
            f.write(f'region = {region}\n')
            f.write('output = json\n')
        
        print("✅ AWS credentials saved to ~/.aws/credentials")
        
    except Exception as e:
        print(f"⚠️ Could not save to AWS config file: {e}")

def check_aws_status():
    """Check current AWS configuration status"""
    print("Checking AWS Configuration Status...")
    print("-" * 35)
    
    # Check environment variables
    env_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    env_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    env_region = os.environ.get('AWS_DEFAULT_REGION')
    
    print(f"Environment Variables:")
    print(f"  AWS_ACCESS_KEY_ID: {'✅ Set' if env_access_key else '❌ Not set'}")
    print(f"  AWS_SECRET_ACCESS_KEY: {'✅ Set' if env_secret_key else '❌ Not set'}")
    print(f"  AWS_DEFAULT_REGION: {env_region or '❌ Not set'}")
    
    # Check AWS credentials file
    credentials_file = os.path.expanduser('~/.aws/credentials')
    config_file = os.path.expanduser('~/.aws/config')
    
    print(f"\nAWS Config Files:")
    print(f"  ~/.aws/credentials: {'✅ Exists' if os.path.exists(credentials_file) else '❌ Not found'}")
    print(f"  ~/.aws/config: {'✅ Exists' if os.path.exists(config_file) else '❌ Not found'}")
    
    # Test AWS connectivity
    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        
        print(f"\nAWS Connection Test:")
        print(f"  Status: ✅ Connected")
        print(f"  Account ID: {identity['Account']}")
        print(f"  User: {identity['Arn'].split('/')[-1]}")
        
        return True
        
    except NoCredentialsError:
        print(f"\nAWS Connection Test:")
        print(f"  Status: ❌ No credentials configured")
        return False
        
    except Exception as e:
        print(f"\nAWS Connection Test:")
        print(f"  Status: ❌ Connection failed: {e}")
        return False

def main():
    """Main configuration function"""
    print("🔧 AWS Configuration Tool")
    print("=" * 50)
    
    # Check current status
    if check_aws_status():
        print("\n✅ AWS is already configured and working!")
        
        reconfigure = input("\nReconfigure AWS credentials? (y/n): ").lower().strip()
        if reconfigure != 'y':
            print("Configuration unchanged.")
            return
    
    print("\n🔑 Configure AWS Credentials:")
    print("Get your credentials from: https://console.aws.amazon.com/iam/home#/security_credentials")
    
    if configure_aws_credentials():
        print("\n🎉 AWS configuration complete!")
        print("Now restart your application to use AWS services.")
        
        # Test cloud engine
        print("\n🧪 Testing cloud engine...")
        try:
            from cloud_engine import CloudEngine
            engine = CloudEngine()
            if engine.aws_enabled:
                print("✅ Cloud engine now using AWS!")
            else:
                print("❌ Cloud engine still in local mode. Restart the application.")
        except Exception as e:
            print(f"⚠️ Could not test cloud engine: {e}")
    
    else:
        print("\n❌ AWS configuration failed!")
        print("Please check your credentials and try again.")

if __name__ == "__main__":
    main()