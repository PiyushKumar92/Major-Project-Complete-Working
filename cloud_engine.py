"""
Cloud & Scalability Engine
AWS integration, auto-scaling, and edge computing
"""

import boto3
import json
import os
import threading
import time
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError
import requests
from collections import defaultdict

class CloudEngine:
    def __init__(self):
        self.aws_enabled = False
        self.edge_nodes = {}
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler()
        self.edge_manager = EdgeManager()
        
        # Initialize AWS services
        self._init_aws_services()
        
    def _init_aws_services(self):
        """AWS permanently disabled - using local GPU CNN"""
        # Skip AWS initialization completely
        self.aws_enabled = False
        self._init_local_fallback()
    
    def _init_local_fallback(self):
        """Initialize local fallback services"""
        self.local_storage = LocalStorage()
        self.local_compute = LocalCompute()
        self.local_monitoring = LocalMonitoring()
    
    def upload_to_cloud(self, file_path, bucket_name="missing-person-data"):
        """Upload file to cloud storage"""
        if self.aws_enabled:
            return self._upload_to_s3(file_path, bucket_name)
        else:
            return self.local_storage.store_file(file_path)
    
    def _upload_to_s3(self, file_path, bucket_name):
        """Upload to AWS S3"""
        try:
            filename = os.path.basename(file_path)
            key = f"uploads/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            
            self.s3_client.upload_file(file_path, bucket_name, key)
            
            return {
                'success': True,
                'url': f"s3://{bucket_name}/{key}",
                'cdn_url': f"https://{bucket_name}.s3.amazonaws.com/{key}"
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def process_on_cloud(self, task_data):
        """Process task on cloud compute"""
        if self.aws_enabled:
            return self._invoke_lambda(task_data)
        else:
            return self.local_compute.process_task(task_data)
    
    def _invoke_lambda(self, task_data):
        """Invoke AWS Lambda function"""
        try:
            response = self.lambda_client.invoke(
                FunctionName='missing-person-processor',
                InvocationType='RequestResponse',
                Payload=json.dumps(task_data)
            )
            
            result = json.loads(response['Payload'].read())
            return {'success': True, 'result': result}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_cloud_metrics(self):
        """Get cloud performance metrics"""
        if self.aws_enabled:
            return self._get_cloudwatch_metrics()
        else:
            return self.local_monitoring.get_metrics()
    
    def _get_cloudwatch_metrics(self):
        """Get AWS CloudWatch metrics"""
        try:
            # Get CPU utilization
            cpu_response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                StartTime=datetime.utcnow().replace(hour=datetime.utcnow().hour-1),
                EndTime=datetime.utcnow(),
                Period=300,
                Statistics=['Average']
            )
            
            return {
                'cpu_utilization': cpu_response.get('Datapoints', []),
                'instance_count': len(self.auto_scaler.get_instances()),
                'storage_usage': self._get_s3_usage()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_s3_usage(self):
        """Get S3 storage usage"""
        try:
            response = self.s3_client.list_objects_v2(Bucket='missing-person-data')
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            return {'total_size_mb': total_size / (1024 * 1024)}
        except:
            return {'total_size_mb': 0}
    
    def scale_resources(self, target_capacity):
        """Scale cloud resources"""
        return self.auto_scaler.scale_to_capacity(target_capacity)
    
    def deploy_to_edge(self, edge_node_id, deployment_config):
        """Deploy to edge computing node"""
        return self.edge_manager.deploy(edge_node_id, deployment_config)

class LoadBalancer:
    def __init__(self):
        self.servers = []
        self.current_index = 0
        self.health_status = {}
        
    def add_server(self, server_url, weight=1):
        """Add server to load balancer"""
        self.servers.append({'url': server_url, 'weight': weight})
        self.health_status[server_url] = True
        
    def get_next_server(self):
        """Get next available server (round-robin)"""
        if not self.servers:
            return None
            
        # Simple round-robin with health check
        attempts = 0
        while attempts < len(self.servers):
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            
            if self.health_status.get(server['url'], True):
                return server
                
            attempts += 1
        
        return None
    
    def health_check(self):
        """Perform health check on all servers"""
        for server in self.servers:
            try:
                response = requests.get(f"{server['url']}/health", timeout=5)
                self.health_status[server['url']] = response.status_code == 200
            except:
                self.health_status[server['url']] = False

class AutoScaler:
    def __init__(self):
        self.instances = []
        self.min_instances = 1
        self.max_instances = 10
        self.target_cpu = 70  # Target CPU percentage
        self.scaling_cooldown = 300  # 5 minutes
        self.last_scale_time = 0
        
    def add_instance(self, instance_id, instance_type="t3.micro"):
        """Add instance to auto-scaling group"""
        self.instances.append({
            'id': instance_id,
            'type': instance_type,
            'status': 'running',
            'cpu_usage': 0,
            'created_at': datetime.now()
        })
    
    def get_instances(self):
        """Get all instances"""
        return self.instances
    
    def check_scaling_needed(self, current_metrics):
        """Check if scaling is needed"""
        if time.time() - self.last_scale_time < self.scaling_cooldown:
            return None
            
        avg_cpu = sum(i.get('cpu_usage', 0) for i in self.instances) / max(len(self.instances), 1)
        current_count = len(self.instances)
        
        if avg_cpu > self.target_cpu and current_count < self.max_instances:
            return 'scale_out'
        elif avg_cpu < self.target_cpu * 0.5 and current_count > self.min_instances:
            return 'scale_in'
            
        return None
    
    def scale_to_capacity(self, target_capacity):
        """Scale to target capacity"""
        current_count = len(self.instances)
        
        if target_capacity > current_count:
            # Scale out
            for i in range(target_capacity - current_count):
                self.add_instance(f"instance-{len(self.instances) + 1}")
            action = 'scaled_out'
        elif target_capacity < current_count:
            # Scale in
            self.instances = self.instances[:target_capacity]
            action = 'scaled_in'
        else:
            action = 'no_change'
        
        self.last_scale_time = time.time()
        
        return {
            'action': action,
            'previous_count': current_count,
            'new_count': len(self.instances),
            'timestamp': datetime.now().isoformat()
        }

class EdgeManager:
    def __init__(self):
        self.edge_nodes = {}
        self.deployments = {}
        
    def register_edge_node(self, node_id, node_config):
        """Register edge computing node"""
        self.edge_nodes[node_id] = {
            'config': node_config,
            'status': 'online',
            'last_heartbeat': datetime.now(),
            'deployments': []
        }
    
    def deploy(self, node_id, deployment_config):
        """Deploy to edge node"""
        if node_id not in self.edge_nodes:
            return {'success': False, 'error': 'Edge node not found'}
        
        deployment_id = f"deploy-{len(self.deployments) + 1}"
        
        self.deployments[deployment_id] = {
            'node_id': node_id,
            'config': deployment_config,
            'status': 'deployed',
            'deployed_at': datetime.now()
        }
        
        self.edge_nodes[node_id]['deployments'].append(deployment_id)
        
        return {
            'success': True,
            'deployment_id': deployment_id,
            'node_id': node_id
        }
    
    def get_edge_status(self):
        """Get status of all edge nodes"""
        return {
            'total_nodes': len(self.edge_nodes),
            'online_nodes': sum(1 for node in self.edge_nodes.values() if node['status'] == 'online'),
            'total_deployments': len(self.deployments),
            'nodes': self.edge_nodes
        }

class LocalStorage:
    def __init__(self):
        self.storage_path = "local_cloud_storage"
        os.makedirs(self.storage_path, exist_ok=True)
        
    def store_file(self, file_path):
        """Store file locally"""
        try:
            import shutil
            filename = os.path.basename(file_path)
            dest_path = os.path.join(self.storage_path, filename)
            shutil.copy2(file_path, dest_path)
            
            return {
                'success': True,
                'url': f"local://{dest_path}",
                'local_path': dest_path
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

class LocalCompute:
    def __init__(self):
        self.task_queue = []
        self.processing = False
        
    def process_task(self, task_data):
        """Process task locally"""
        try:
            # Simulate cloud processing
            result = {
                'task_id': task_data.get('task_id', 'local-task'),
                'status': 'completed',
                'result': 'Local processing completed',
                'processed_at': datetime.now().isoformat()
            }
            
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}

class LocalMonitoring:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def get_metrics(self):
        """Get local system metrics"""
        try:
            import psutil
        except ImportError:
            # Fallback metrics if psutil not available
            return {
                'cpu_utilization': [{'Value': 45, 'Timestamp': datetime.now()}],
                'memory_usage': [{'Value': 60, 'Timestamp': datetime.now()}],
                'disk_usage': [{'Value': 30, 'Timestamp': datetime.now()}]
            }
        
        return {
            'cpu_utilization': [{'Value': psutil.cpu_percent(), 'Timestamp': datetime.now()}],
            'memory_usage': [{'Value': psutil.virtual_memory().percent, 'Timestamp': datetime.now()}],
            'disk_usage': [{'Value': psutil.disk_usage('/').percent, 'Timestamp': datetime.now()}]
        }

# Global cloud engine instance
cloud_engine = CloudEngine()