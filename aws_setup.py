"""
AWS Setup and Configuration
Complete AWS integration setup for production use
"""

import boto3
import json
import os
from botocore.exceptions import ClientError, NoCredentialsError

class AWSSetup:
    def __init__(self):
        self.region = 'us-east-1'
        self.account_id = None
        self.setup_status = {}
        
    def configure_aws_credentials(self, access_key, secret_key, region='us-east-1'):
        """Configure AWS credentials programmatically"""
        try:
            # Set environment variables
            os.environ['AWS_ACCESS_KEY_ID'] = access_key
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
            os.environ['AWS_DEFAULT_REGION'] = region
            
            self.region = region
            
            # Test credentials
            sts = boto3.client('sts', region_name=region)
            identity = sts.get_caller_identity()
            self.account_id = identity['Account']
            
            print(f"✓ AWS credentials configured successfully")
            print(f"   Account ID: {self.account_id}")
            print(f"   Region: {region}")
            
            return {'success': True, 'account_id': self.account_id}
            
        except Exception as e:
            print(f"❌ AWS credential configuration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_s3_bucket(self, bucket_name='missing-person-system-data'):
        """Create S3 bucket for data storage"""
        try:
            s3 = boto3.client('s3', region_name=self.region)
            
            # Create bucket
            if self.region == 'us-east-1':
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            
            # Configure bucket policy
            bucket_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "MissingPersonSystemAccess",
                        "Effect": "Allow",
                        "Principal": {"AWS": f"arn:aws:iam::{self.account_id}:root"},
                        "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
                        "Resource": f"arn:aws:s3:::{bucket_name}/*"
                    }
                ]
            }
            
            s3.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(bucket_policy)
            )
            
            print(f"✓ S3 bucket created: {bucket_name}")
            self.setup_status['s3_bucket'] = bucket_name
            
            return {'success': True, 'bucket_name': bucket_name}
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"✓ S3 bucket already exists: {bucket_name}")
                self.setup_status['s3_bucket'] = bucket_name
                return {'success': True, 'bucket_name': bucket_name}
            else:
                print(f"❌ S3 bucket creation failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def create_lambda_function(self):
        """Create Lambda function for processing"""
        try:
            lambda_client = boto3.client('lambda', region_name=self.region)
            
            # Lambda function code
            lambda_code = '''
import json
import boto3
import base64
from io import BytesIO

def lambda_handler(event, context):
    try:
        task_type = event.get('task_type', 'unknown')
        
        if task_type == 'face_recognition':
            return process_face_recognition(event)
        elif task_type == 'image_analysis':
            return process_image_analysis(event)
        elif task_type == 'video_processing':
            return process_video(event)
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Unknown task type'})
            }
            
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def process_face_recognition(event):
    # AWS Rekognition integration
    rekognition = boto3.client('rekognition')
    
    image_data = event.get('image_data')
    if image_data:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Detect faces
        response = rekognition.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'faces_detected': len(response['FaceDetails']),
                'faces': response['FaceDetails']
            })
        }
    
    return {
        'statusCode': 400,
        'body': json.dumps({'error': 'No image data provided'})
    }

def process_image_analysis(event):
    # AWS Rekognition for object detection
    rekognition = boto3.client('rekognition')
    
    image_data = event.get('image_data')
    if image_data:
        image_bytes = base64.b64decode(image_data)
        
        # Detect labels
        response = rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=20,
            MinConfidence=70
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'labels': response['Labels']
            })
        }
    
    return {
        'statusCode': 400,
        'body': json.dumps({'error': 'No image data provided'})
    }

def process_video(event):
    # Video processing placeholder
    return {
        'statusCode': 200,
        'body': json.dumps({
            'status': 'video_processing_started',
            'job_id': 'mock-job-id'
        })
    }
'''
            
            # Create Lambda function
            response = lambda_client.create_function(
                FunctionName='missing-person-processor',
                Runtime='python3.9',
                Role=f'arn:aws:iam::{self.account_id}:role/MissingPersonLambdaRole',
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': lambda_code.encode()},
                Description='Missing Person System AI Processing',
                Timeout=300,
                MemorySize=512
            )
            
            print(f"✓ Lambda function created: missing-person-processor")
            self.setup_status['lambda_function'] = response['FunctionArn']
            
            return {'success': True, 'function_arn': response['FunctionArn']}
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                print(f"✓ Lambda function already exists")
                return {'success': True, 'message': 'Function already exists'}
            else:
                print(f"❌ Lambda function creation failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def create_iam_roles(self):
        """Create necessary IAM roles"""
        try:
            iam = boto3.client('iam', region_name=self.region)
            
            # Lambda execution role
            lambda_trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "lambda.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create Lambda role
            try:
                lambda_role = iam.create_role(
                    RoleName='MissingPersonLambdaRole',
                    AssumeRolePolicyDocument=json.dumps(lambda_trust_policy),
                    Description='Lambda execution role for Missing Person System'
                )
                print(f"✓ Lambda IAM role created")
            except ClientError as e:
                if e.response['Error']['Code'] == 'EntityAlreadyExists':
                    print(f"✓ Lambda IAM role already exists")
                else:
                    raise e
            
            # Attach policies to Lambda role
            policies = [
                'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                'arn:aws:iam::aws:policy/AmazonRekognitionFullAccess',
                'arn:aws:iam::aws:policy/AmazonS3FullAccess'
            ]
            
            for policy in policies:
                try:
                    iam.attach_role_policy(
                        RoleName='MissingPersonLambdaRole',
                        PolicyArn=policy
                    )
                except ClientError:
                    pass  # Policy might already be attached
            
            # ECS task execution role
            ecs_trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            try:
                ecs_role = iam.create_role(
                    RoleName='MissingPersonECSRole',
                    AssumeRolePolicyDocument=json.dumps(ecs_trust_policy),
                    Description='ECS task execution role for Missing Person System'
                )
                print(f"✓ ECS IAM role created")
            except ClientError as e:
                if e.response['Error']['Code'] == 'EntityAlreadyExists':
                    print(f"✓ ECS IAM role already exists")
                else:
                    raise e
            
            # Attach ECS policy
            iam.attach_role_policy(
                RoleName='MissingPersonECSRole',
                PolicyArn='arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy'
            )
            
            self.setup_status['iam_roles'] = ['MissingPersonLambdaRole', 'MissingPersonECSRole']
            
            return {'success': True, 'roles_created': 2}
            
        except Exception as e:
            print(f"❌ IAM role creation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_ecs_cluster(self):
        """Create ECS cluster for containerized services"""
        try:
            ecs = boto3.client('ecs', region_name=self.region)
            
            # Create ECS cluster
            response = ecs.create_cluster(
                clusterName='missing-person-cluster',
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ]
            )
            
            print(f"✓ ECS cluster created: missing-person-cluster")
            self.setup_status['ecs_cluster'] = response['cluster']['clusterArn']
            
            return {'success': True, 'cluster_arn': response['cluster']['clusterArn']}
            
        except ClientError as e:
            if 'already exists' in str(e):
                print(f"✓ ECS cluster already exists")
                return {'success': True, 'message': 'Cluster already exists'}
            else:
                print(f"❌ ECS cluster creation failed: {e}")
                return {'success': False, 'error': str(e)}
    
    def setup_cloudwatch_monitoring(self):
        """Setup CloudWatch monitoring"""
        try:
            cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # Create custom metrics
            metrics = [
                {
                    'MetricName': 'MissingPersonDetections',
                    'Namespace': 'MissingPersonSystem',
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'ProcessingLatency',
                    'Namespace': 'MissingPersonSystem', 
                    'Unit': 'Milliseconds'
                }
            ]
            
            # Create alarms
            cloudwatch.put_metric_alarm(
                AlarmName='HighProcessingLatency',
                ComparisonOperator='GreaterThanThreshold',
                EvaluationPeriods=2,
                MetricName='ProcessingLatency',
                Namespace='MissingPersonSystem',
                Period=300,
                Statistic='Average',
                Threshold=5000.0,
                ActionsEnabled=True,
                AlarmDescription='Alert when processing latency is high'
            )
            
            print(f"✓ CloudWatch monitoring configured")
            self.setup_status['cloudwatch'] = True
            
            return {'success': True, 'alarms_created': 1}
            
        except Exception as e:
            print(f"❌ CloudWatch setup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def deploy_full_infrastructure(self, access_key, secret_key, region='us-east-1'):
        """Deploy complete AWS infrastructure"""
        print("🚀 Starting AWS infrastructure deployment...")
        
        # Step 1: Configure credentials
        cred_result = self.configure_aws_credentials(access_key, secret_key, region)
        if not cred_result['success']:
            return cred_result
        
        # Step 2: Create IAM roles (must be first)
        print("\n📋 Creating IAM roles...")
        iam_result = self.create_iam_roles()
        
        # Wait for IAM roles to propagate
        import time
        time.sleep(10)
        
        # Step 3: Create S3 bucket
        print("\n🪣 Creating S3 bucket...")
        s3_result = self.create_s3_bucket()
        
        # Step 4: Create Lambda function
        print("\n⚡ Creating Lambda function...")
        lambda_result = self.create_lambda_function()
        
        # Step 5: Create ECS cluster
        print("\n🐳 Creating ECS cluster...")
        ecs_result = self.create_ecs_cluster()
        
        # Step 6: Setup monitoring
        print("\n📊 Setting up CloudWatch monitoring...")
        cw_result = self.setup_cloudwatch_monitoring()
        
        # Summary
        print("\n" + "="*50)
        print("🎉 AWS Infrastructure Deployment Complete!")
        print("="*50)
        
        success_count = sum([
            cred_result['success'],
            iam_result['success'],
            s3_result['success'],
            lambda_result['success'],
            ecs_result['success'],
            cw_result['success']
        ])
        
        print(f"✅ Services deployed: {success_count}/6")
        print(f"🌍 Region: {self.region}")
        print(f"🏢 Account ID: {self.account_id}")
        
        if 's3_bucket' in self.setup_status:
            print(f"🪣 S3 Bucket: {self.setup_status['s3_bucket']}")
        
        if 'lambda_function' in self.setup_status:
            print(f"⚡ Lambda Function: missing-person-processor")
        
        if 'ecs_cluster' in self.setup_status:
            print(f"🐳 ECS Cluster: missing-person-cluster")
        
        print("\n🔧 Next Steps:")
        print("1. Update cloud_engine.py with your bucket name")
        print("2. Test the system with: python test_cloud.py")
        print("3. Access dashboard: http://localhost:5000/cloud-dashboard")
        
        return {
            'success': success_count == 6,
            'deployed_services': success_count,
            'setup_status': self.setup_status,
            'account_id': self.account_id,
            'region': self.region
        }
    
    def get_setup_status(self):
        """Get current setup status"""
        return {
            'account_id': self.account_id,
            'region': self.region,
            'setup_status': self.setup_status
        }

# Global AWS setup instance
aws_setup = AWSSetup()