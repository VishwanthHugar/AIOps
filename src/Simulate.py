import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import uuid
import threading
import queue
import os
import csv
import pandas as pd

#import sqlite3

def json_serializer(obj):
    """JSON serializer function that handles datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class TelecomKPIGenerator:
    """
    Generates simulated telecom KPI data including:
    - Network Performance Metrics (latency, throughput, packet loss, jitter)
    - Service Quality Indicators (call setup success rate, data session success rate)
    - Resource Utilization (CPU, memory, bandwidth utilization)
    - Availability Metrics (uptime, downtime events)
    """
    
    def __init__(self):
        self.cell_ids = [f"CELL_{i:04d}" for i in range(1, 101)]  # 100 cell sites
        self.base_stations = [f"BS_{i:03d}" for i in range(1, 21)]  # 20 base stations
        
    def generate_network_performance_kpis(self, timestamp, cell_id):
        """Generate network performance KPIs"""
        # Add some realistic variations and anomaly patterns
        base_latency = random.normalvariate(15, 3)  # ms
        base_throughput = random.normalvariate(150, 30)  # Mbps
        base_packet_loss = random.normalvariate(0.5, 0.2)  # %
        base_jitter = random.normalvariate(2, 0.5)  # ms
        
        # Introduce occasional anomalies
        if random.random() < 0.05:  # 5% chance of anomaly
            base_latency *= random.uniform(2, 5)  # High latency
            base_packet_loss *= random.uniform(3, 10)  # High packet loss
            
        return {
            'timestamp': timestamp,
            'cell_id': cell_id,
            'metric_type': 'network_performance',
            'latency_ms': max(0, base_latency),
            'throughput_mbps': max(0, base_throughput),
            'packet_loss_percent': max(0, min(100, base_packet_loss)),
            'jitter_ms': max(0, base_jitter),
            'rtt_ms': max(0, base_latency * 2 + random.normalvariate(0, 1))
        }
    
    def generate_service_quality_kpis(self, timestamp, cell_id):
        """Generate service quality KPIs"""
        base_cssr = random.normalvariate(98.5, 1.5)  # Call Setup Success Rate %
        base_dssr = random.normalvariate(97.8, 2.0)  # Data Session Success Rate %
        base_handover_success = random.normalvariate(99.2, 0.8)  # Handover Success Rate %
        
        # Introduce anomalies
        if random.random() < 0.03:  # 3% chance of service degradation
            base_cssr *= random.uniform(0.7, 0.9)
            base_dssr *= random.uniform(0.6, 0.8)
            
        return {
            'timestamp': timestamp,
            'cell_id': cell_id,
            'metric_type': 'service_quality',
            'call_setup_success_rate': max(0, min(100, base_cssr)),
            'data_session_success_rate': max(0, min(100, base_dssr)),
            'handover_success_rate': max(0, min(100, base_handover_success)),
            'call_drop_rate': max(0, min(10, random.normalvariate(0.3, 0.1)))
        }
    
    def generate_resource_utilization_kpis(self, timestamp, cell_id):
        """Generate resource utilization KPIs"""
        base_cpu = random.normalvariate(45, 15)  # CPU %
        base_memory = random.normalvariate(60, 20)  # Memory %
        base_bandwidth = random.normalvariate(70, 25)  # Bandwidth %
        
        # High utilization anomalies
        if random.random() < 0.04:  # 4% chance of high utilization
            base_cpu = random.uniform(85, 98)
            base_memory = random.uniform(80, 95)
            
        return {
            'timestamp': timestamp,
            'cell_id': cell_id,
            'metric_type': 'resource_utilization',
            'cpu_utilization_percent': max(0, min(100, base_cpu)),
            'memory_utilization_percent': max(0, min(100, base_memory)),
            'bandwidth_utilization_percent': max(0, min(100, base_bandwidth)),
            'disk_utilization_percent': max(0, min(100, random.normalvariate(30, 10)))
        }

class TelecomAlarmGenerator:
    """
    Generates simulated telecom alarms with different severity levels
    """
    
    def __init__(self):
        self.alarm_types = [
            'HIGH_CPU_UTILIZATION', 'HIGH_MEMORY_USAGE', 'NETWORK_CONNECTIVITY_LOSS',
            'PACKET_LOSS_THRESHOLD', 'HIGH_LATENCY', 'SERVICE_DEGRADATION',
            'HARDWARE_FAILURE', 'POWER_OUTAGE', 'TEMPERATURE_ALERT',
            'BANDWIDTH_CONGESTION', 'AUTHENTICATION_FAILURE', 'SECURITY_BREACH'
        ]
        
        self.severity_levels = ['CRITICAL', 'MAJOR', 'MINOR', 'WARNING', 'INFO']
        
        self.alarm_descriptions = {
            'HIGH_CPU_UTILIZATION': 'CPU utilization exceeded threshold',
            'HIGH_MEMORY_USAGE': 'Memory usage above acceptable limits',
            'NETWORK_CONNECTIVITY_LOSS': 'Network connectivity lost to remote node',
            'PACKET_LOSS_THRESHOLD': 'Packet loss rate exceeds configured threshold',
            'HIGH_LATENCY': 'Network latency above SLA requirements',
            'SERVICE_DEGRADATION': 'Service quality metrics below threshold',
            'HARDWARE_FAILURE': 'Hardware component failure detected',
            'POWER_OUTAGE': 'Power supply interruption detected',
            'TEMPERATURE_ALERT': 'Equipment temperature outside normal range',
            'BANDWIDTH_CONGESTION': 'Bandwidth utilization at capacity',
            'AUTHENTICATION_FAILURE': 'Authentication attempt failed',
            'SECURITY_BREACH': 'Potential security breach detected'
        }
    
    def generate_alarm(self, timestamp, cell_id=None):
        """Generate a telecom alarm"""
        alarm_type = random.choice(self.alarm_types)
        
        # Weight severity probabilities (more warnings/minor, fewer critical)
        severity_weights = {'CRITICAL': 0.05, 'MAJOR': 0.15, 'MINOR': 0.30, 'WARNING': 0.35, 'INFO': 0.15}
        severity = np.random.choice(list(severity_weights.keys()), p=list(severity_weights.values()))
        
        if not cell_id:
            cell_id = random.choice([f"CELL_{i:04d}" for i in range(1, 101)])
            
        return {
            'timestamp': timestamp,
            'alarm_id': str(uuid.uuid4()),
            'cell_id': cell_id,
            'alarm_type': alarm_type,
            'severity': severity,
            'description': self.alarm_descriptions[alarm_type],
            'status': random.choice(['ACTIVE', 'CLEARED', 'ACKNOWLEDGED']),
            'source_system': f"NMS_{random.randint(1, 5)}",
            'affected_services': random.randint(1, 10) if severity in ['CRITICAL', 'MAJOR'] else 0
        }

class TelecomLogGenerator:
    """
    Generates simulated telecom system logs
    """
    
    def __init__(self):
        self.log_levels = ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL']
        self.components = [
            'RAN_CONTROLLER', 'BASE_STATION', 'CORE_NETWORK', 'PACKET_GATEWAY',
            'MME', 'HSS', 'PCRF', 'IMS', 'BILLING_SYSTEM', 'NMS'
        ]
        
        self.log_messages = {
            'DEBUG': ['Debug trace for component initialization', 'Debug: Processing routine maintenance task'],
            'INFO': ['User session established successfully', 'Handover completed', 'System backup completed'],
            'WARN': ['Resource utilization approaching threshold', 'Retry attempt for failed operation', 'Configuration change detected'],
            'ERROR': ['Failed to establish connection', 'Authentication timeout', 'Database query failed'],
            'FATAL': ['System component crashed', 'Critical system failure', 'Emergency shutdown initiated']
        }
    
    def generate_log_entry(self, timestamp, component=None):
        """Generate a log entry"""
        # Weight log level probabilities (more INFO/DEBUG, fewer FATAL)
        level_weights = {'DEBUG': 0.25, 'INFO': 0.40, 'WARN': 0.20, 'ERROR': 0.12, 'FATAL': 0.03}
        log_level = np.random.choice(list(level_weights.keys()), p=list(level_weights.values()))
        
        if not component:
            component = random.choice(self.components)
            
        message = random.choice(self.log_messages[log_level])
        
        return {
            'timestamp': timestamp,
            'log_id': str(uuid.uuid4()),
            'component': component,
            'log_level': log_level,
            'message': message,
            'thread_id': f"T_{random.randint(1000, 9999)}",
            'session_id': f"S_{random.randint(100000, 999999)}",
            'user_id': f"U_{random.randint(1, 10000)}",
            'ip_address': f"{random.randint(192, 192)}.{random.randint(168, 168)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        }



class DataPipelineCollector:
    """
    Main data pipeline collector that simulates real-time data collection
    from multiple telecom data sources (KPIs, Alarms, Logs)
    Now saves data to CSV files partitioned by time of run.
    """
    def __init__(self, output_dir='../test_data'): #../output_data
        self.is_running = False
        self.kpi_queue = queue.Queue(maxsize=1000)
        self.alarm_queue = queue.Queue(maxsize=500)
        self.log_queue = queue.Queue(maxsize=2000)
        self.kpi_generator = TelecomKPIGenerator()
        self.alarm_generator = TelecomAlarmGenerator()
        self.log_generator = TelecomLogGenerator()

        # Partitioned output directory by run time
        self.run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'run_{self.run_time}')
        os.makedirs(self.output_dir, exist_ok=True)
        print (f"Output directory created: {self.output_dir}")
        # File paths
        self.kpi_csv = os.path.join(self.output_dir, 'kpis.csv')
        self.alarm_csv = os.path.join(self.output_dir, 'alarms.csv')
        self.log_csv = os.path.join(self.output_dir, 'logs.csv')

        # Write headers
        self._write_csv_headers()

    def _write_csv_headers(self):
        # Write headers for each CSV file
        if not os.path.exists(self.kpi_csv):
            with open(self.kpi_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'cell_id', 'metric_type', 'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
                    'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate', 'call_drop_rate',
                    'cpu_utilization_percent', 'memory_utilization_percent', 'bandwidth_utilization_percent', 'disk_utilization_percent'])
                writer.writeheader()
        if not os.path.exists(self.alarm_csv):
            with open(self.alarm_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'alarm_id', 'cell_id', 'alarm_type', 'severity', 'description', 'status', 'source_system', 'affected_services'])
                writer.writeheader()
        if not os.path.exists(self.log_csv):
            with open(self.log_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'log_id', 'component', 'log_level', 'message', 'thread_id', 'session_id', 'user_id', 'ip_address'])
                writer.writeheader()
    
    def collect_kpis(self):
        """Simulate real-time KPI collection"""
        while self.is_running:
            try:
                current_time = datetime.now()
                # Simulate collecting from multiple cells
                selected_cells = random.sample(self.kpi_generator.cell_ids, random.randint(3, 8))
                
                for cell_id in selected_cells:
                    # Generate different KPI types
                    kpi_types = ['network_performance', 'service_quality', 'resource_utilization']
                    for kpi_type in random.sample(kpi_types, random.randint(1, 3)):
                        if kpi_type == 'network_performance':
                            kpi_data = self.kpi_generator.generate_network_performance_kpis(current_time, cell_id)
                        elif kpi_type == 'service_quality':
                            kpi_data = self.kpi_generator.generate_service_quality_kpis(current_time, cell_id)
                        else:
                            kpi_data = self.kpi_generator.generate_resource_utilization_kpis(current_time, cell_id)
                        
                        self.kpi_queue.put(kpi_data)
                
                time.sleep(10)  # Collect KPIs every 10 seconds
                
            except Exception as e:
                print(f"Error in KPI collection: {e}")
    
    def collect_alarms(self):
        """Simulate real-time alarm collection"""
        while self.is_running:
            try:
                current_time = datetime.now()
                # Always generate alarms every interval
                num_alarms = random.choices([2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.1])[0]
                for _ in range(num_alarms):
                    alarm_data = self.alarm_generator.generate_alarm(current_time)
                    self.alarm_queue.put(alarm_data)
                time.sleep(10)  # Check for alarms every 10 seconds
            except Exception as e:
                print(f"Error in alarm collection: {e}")
    
    def collect_logs(self):
        """Simulate real-time log collection"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Generate multiple log entries
                num_logs = random.randint(2, 6)
                
                for _ in range(num_logs):
                    log_data = self.log_generator.generate_log_entry(current_time)
                    self.log_queue.put(log_data)
                
                time.sleep(5)  # Collect logs every 5 seconds
                
            except Exception as e:
                print(f"Error in log collection: {e}")
    
    def process_and_store_data(self):
        """Process data from queues and store in CSV files"""
        while self.is_running:
            try:
                # Process KPIs
                kpis_processed = 0
                kpi_rows = []
                while not self.kpi_queue.empty() and kpis_processed < 50:
                    try:
                        kpi_data = self.kpi_queue.get_nowait()
                        # Flatten all possible fields for CSV
                        row = {
                            'timestamp': kpi_data.get('timestamp'),
                            'cell_id': kpi_data.get('cell_id'),
                            'metric_type': kpi_data.get('metric_type'),
                            'latency_ms': kpi_data.get('latency_ms'),
                            'throughput_mbps': kpi_data.get('throughput_mbps'),
                            'packet_loss_percent': kpi_data.get('packet_loss_percent'),
                            'jitter_ms': kpi_data.get('jitter_ms'),
                            'rtt_ms': kpi_data.get('rtt_ms'),
                            'call_setup_success_rate': kpi_data.get('call_setup_success_rate'),
                            'data_session_success_rate': kpi_data.get('data_session_success_rate'),
                            'handover_success_rate': kpi_data.get('handover_success_rate'),
                            'call_drop_rate': kpi_data.get('call_drop_rate'),
                            'cpu_utilization_percent': kpi_data.get('cpu_utilization_percent'),
                            'memory_utilization_percent': kpi_data.get('memory_utilization_percent'),
                            'bandwidth_utilization_percent': kpi_data.get('bandwidth_utilization_percent'),
                            'disk_utilization_percent': kpi_data.get('disk_utilization_percent'),
                        }
                        kpi_rows.append(row)
                        kpis_processed += 1
                    except queue.Empty:
                        break
                if kpi_rows:
                    with open(self.kpi_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=kpi_rows[0].keys())
                        writer.writerows(kpi_rows)

                # Process Alarms
                alarms_processed = 0
                alarm_rows = []
                while not self.alarm_queue.empty() and alarms_processed < 20:
                    try:
                        alarm_data = self.alarm_queue.get_nowait()
                        row = {
                            'timestamp': alarm_data.get('timestamp'),
                            'alarm_id': alarm_data.get('alarm_id'),
                            'cell_id': alarm_data.get('cell_id'),
                            'alarm_type': alarm_data.get('alarm_type'),
                            'severity': alarm_data.get('severity'),
                            'description': alarm_data.get('description'),
                            'status': alarm_data.get('status'),
                            'source_system': alarm_data.get('source_system'),
                            'affected_services': alarm_data.get('affected_services'),
                        }
                        alarm_rows.append(row)
                        alarms_processed += 1
                    except queue.Empty:
                        break
                if alarm_rows:
                    with open(self.alarm_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=alarm_rows[0].keys())
                        writer.writerows(alarm_rows)

                # Process Logs
                logs_processed = 0
                log_rows = []
                while not self.log_queue.empty() and logs_processed < 100:
                    try:
                        log_data = self.log_queue.get_nowait()
                        row = {
                            'timestamp': log_data.get('timestamp'),
                            'log_id': log_data.get('log_id'),
                            'component': log_data.get('component'),
                            'log_level': log_data.get('log_level'),
                            'message': log_data.get('message'),
                            'thread_id': log_data.get('thread_id'),
                            'session_id': log_data.get('session_id'),
                            'user_id': log_data.get('user_id'),
                            'ip_address': log_data.get('ip_address'),
                        }
                        log_rows.append(row)
                        logs_processed += 1
                    except queue.Empty:
                        break
                if log_rows:
                    with open(self.log_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
                        writer.writerows(log_rows)

                if kpis_processed > 0 or alarms_processed > 0 or logs_processed > 0:
                    print(f"Processed - KPIs: {kpis_processed}, Alarms: {alarms_processed}, Logs: {logs_processed}")

                time.sleep(2)  # Process every 2 seconds
            except Exception as e:
                print(f"Error in data processing: {e}")
    
    def start_collection(self):
        """Start the data collection pipeline"""
        self.is_running = True
        
        # Start collection threads
        kpi_thread = threading.Thread(target=self.collect_kpis, daemon=True)
        alarm_thread = threading.Thread(target=self.collect_alarms, daemon=True)
        log_thread = threading.Thread(target=self.collect_logs, daemon=True)
        processor_thread = threading.Thread(target=self.process_and_store_data, daemon=True)
        
        kpi_thread.start()
        alarm_thread.start()
        log_thread.start()
        processor_thread.start()
        
        print("Data Pipeline Collector started successfully!")
        print("Collecting KPIs, Alarms, and Logs in real-time...")
        
        return kpi_thread, alarm_thread, log_thread, processor_thread
    
    def stop_collection(self):
        """Stop the data collection pipeline"""
        self.is_running = False
        print("Data Pipeline Collector stopped.")
    
    def get_data_summary(self):
        """Get summary of collected data from CSV files"""
        summary = {}
        # Total counts
        if os.path.exists(self.kpi_csv):
            kpi_df = pd.read_csv(self.kpi_csv)
            summary['total_kpis'] = len(kpi_df)
            # Recent: last hour
            kpi_df['timestamp'] = pd.to_datetime(kpi_df['timestamp'])
            summary['recent_kpis'] = kpi_df[kpi_df['timestamp'] > (datetime.now() - timedelta(hours=1))].shape[0]
        else:
            summary['total_kpis'] = 0
            summary['recent_kpis'] = 0
        if os.path.exists(self.alarm_csv):
            alarm_df = pd.read_csv(self.alarm_csv)
            summary['total_alarms'] = len(alarm_df)
            alarm_df['timestamp'] = pd.to_datetime(alarm_df['timestamp'])
            summary['recent_alarms'] = alarm_df[alarm_df['timestamp'] > (datetime.now() - timedelta(hours=1))].shape[0]
        else:
            summary['total_alarms'] = 0
            summary['recent_alarms'] = 0
        if os.path.exists(self.log_csv):
            log_df = pd.read_csv(self.log_csv)
            summary['total_logs'] = len(log_df)
            log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
            summary['recent_logs'] = log_df[log_df['timestamp'] > (datetime.now() - timedelta(hours=1))].shape[0]
        else:
            summary['total_logs'] = 0
            summary['recent_logs'] = 0
        return summary
    
    def collect_batch_data(self, kpis_data, alarms_data, logs_data):
        """Collect a batch of data (useful for initial population) and save to CSV"""
        # KPIs
        if kpis_data:
            with open(self.kpi_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'cell_id', 'metric_type', 'latency_ms', 'throughput_mbps', 'packet_loss_percent', 'jitter_ms', 'rtt_ms',
                    'call_setup_success_rate', 'data_session_success_rate', 'handover_success_rate', 'call_drop_rate',
                    'cpu_utilization_percent', 'memory_utilization_percent', 'bandwidth_utilization_percent', 'disk_utilization_percent'])
                for kpi in kpis_data:
                    row = {
                        'timestamp': kpi.get('timestamp'),
                        'cell_id': kpi.get('cell_id'),
                        'metric_type': kpi.get('metric_type'),
                        'latency_ms': kpi.get('latency_ms'),
                        'throughput_mbps': kpi.get('throughput_mbps'),
                        'packet_loss_percent': kpi.get('packet_loss_percent'),
                        'jitter_ms': kpi.get('jitter_ms'),
                        'rtt_ms': kpi.get('rtt_ms'),
                        'call_setup_success_rate': kpi.get('call_setup_success_rate'),
                        'data_session_success_rate': kpi.get('data_session_success_rate'),
                        'handover_success_rate': kpi.get('handover_success_rate'),
                        'call_drop_rate': kpi.get('call_drop_rate'),
                        'cpu_utilization_percent': kpi.get('cpu_utilization_percent'),
                        'memory_utilization_percent': kpi.get('memory_utilization_percent'),
                        'bandwidth_utilization_percent': kpi.get('bandwidth_utilization_percent'),
                        'disk_utilization_percent': kpi.get('disk_utilization_percent'),
                    }
                    writer.writerow(row)
        # Alarms
        if alarms_data:
            with open(self.alarm_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'alarm_id', 'cell_id', 'alarm_type', 'severity', 'description', 'status', 'source_system', 'affected_services'])
                for alarm in alarms_data:
                    row = {
                        'timestamp': alarm.get('timestamp'),
                        'alarm_id': alarm.get('alarm_id'),
                        'cell_id': alarm.get('cell_id'),
                        'alarm_type': alarm.get('alarm_type'),
                        'severity': alarm.get('severity'),
                        'description': alarm.get('description'),
                        'status': alarm.get('status'),
                        'source_system': alarm.get('source_system'),
                        'affected_services': alarm.get('affected_services'),
                    }
                    writer.writerow(row)
        # Logs
        if logs_data:
            with open(self.log_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'log_id', 'component', 'log_level', 'message', 'thread_id', 'session_id', 'user_id', 'ip_address'])
                for log in logs_data:
                    row = {
                        'timestamp': log.get('timestamp'),
                        'log_id': log.get('log_id'),
                        'component': log.get('component'),
                        'log_level': log.get('log_level'),
                        'message': log.get('message'),
                        'thread_id': log.get('thread_id'),
                        'session_id': log.get('session_id'),
                        'user_id': log.get('user_id'),
                        'ip_address': log.get('ip_address'),
                    }
                    writer.writerow(row)
        print(f"Stored {len(kpis_data)} KPIs, {len(alarms_data)} alarms, {len(logs_data)} logs")

# Example usage:
if __name__ == "__main__":
    # Initialize the pipeline collector (CSV output)
    collector = DataPipelineCollector()
    
    # Start real-time collection
    threads = collector.start_collection()
    
    try:
        # Let it run for a while
        time.sleep(60)  # Run for 1 minute
        
        # Get data summary
        summary = collector.get_data_summary()
        print(f"\nData Collection Summary:")
        print(f"Total KPIs: {summary['total_kpis']}")
        print(f"Total Alarms: {summary['total_alarms']}")
        print(f"Total Logs: {summary['total_logs']}")
        print(f"Recent KPIs (last hour): {summary['recent_kpis']}")
        print(f"Recent Alarms (last hour): {summary['recent_alarms']}")
        print(f"Recent Logs (last hour): {summary['recent_logs']}")
        
    finally:
        # Stop collection
        collector.stop_collection()

print("Data Pipeline Collector code is ready!")